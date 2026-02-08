import os
import sys
import torch
import argparse
import pyRDDLGym
import warnings
from pyRDDLGym.core.visualizer.movie import MovieGenerator
from tqdm import tqdm

# Silence specific pyRDDLGym warnings about temporary files
warnings.filterwarnings("ignore", category=UserWarning, module="pyRDDLGym")

# Add the project root to sys.path to allow 'from src.xxx import yyy'
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

# Local Imports
from src.generator import generate_instance
from src.visualizer import TrainRouteVisualizer
from src.wrappers import RDDLDecisionWrapper, PPOAdapter
from src.agent import PPO, Memory, device
from src.logger import RewardLogger

# Argument Parser
parser = argparse.ArgumentParser(description="Train the Scheduler RL Agent")
parser.add_argument("--episodes", type=int, default=5000, help="Maximum number of episodes (target)")
parser.add_argument("--additional_episodes", type=int, default=None, help="Number of additional episodes to run from current checkpoint")
parser.add_argument("--log_interval", type=int, default=20, help="Log interval for rewards")
parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint save interval")
parser.add_argument("--update_timestep", type=int, default=2000, help="PPO update timestep")
parser.add_argument("--force_restart", action="store_true", help="Ignore checkpoints and start fresh")
args = parser.parse_args()

# Configuration
# BASE_PATH is already defined above
RDDL_DIR = os.path.join(BASE_PATH, "rddl")
OUTPUT_DIR = os.path.join(BASE_PATH, "output")
DOMAIN_PATH = os.path.join(RDDL_DIR, "domain.rddl")
INSTANCE_PATH = os.path.join(RDDL_DIR, "instance.rddl")

# Initialize Logger
logger = RewardLogger(OUTPUT_DIR)

# 1. Generate Instance (only if missing or forced)
if args.force_restart or not os.path.exists(INSTANCE_PATH):
    rddl_content = generate_instance(num_trains=3, num_stations=4, horizon=50)
    with open(INSTANCE_PATH, "w") as f:
        f.write(rddl_content)

# 2. Setup Environment
env = pyRDDLGym.make(domain=DOMAIN_PATH, instance=INSTANCE_PATH)
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)
movie_gen = MovieGenerator(VIS_DIR, "train_scheduler", 50)
env.set_visualizer(TrainRouteVisualizer, movie_gen=movie_gen, movie_per_episode=True)

# 3. Apply Wrappers
env = RDDLDecisionWrapper(env)
env = PPOAdapter(env)

# 4. Initialize Agent
state_dim = env.observation_space.shape[0]
action_dim = 11
ppo = PPO(
    state_dim,
    action_dim,
    lr=0.002,
    betas=(0.9, 0.999),
    gamma=0.99,
    K_epochs=4,
    eps_clip=0.2,
)
memory = Memory()

# 5. Training Configuration
checkpoint_dir = os.path.join(BASE_PATH, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
latest_model_path = os.path.join(checkpoint_dir, "latest_model.pth")
best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

start_episode = 1
max_episodes = args.episodes
update_timestep = args.update_timestep
log_interval = args.log_interval
save_interval = args.save_interval
time_step = 0
running_reward = 0
best_reward = -float("inf")

if not args.force_restart and os.path.exists(latest_model_path):
    start_episode, best_reward, time_step = ppo.load_checkpoint(latest_model_path)
    start_episode += 1
    print(f"Resuming from episode {start_episode} (Best Reward: {best_reward:.2f})")
else:
    logger.clear_log()

# Adjust max_episodes if additional_episodes is specified
if args.additional_episodes is not None:
    max_episodes = start_episode + args.additional_episodes - 1

# 6. Training Loop
print(f"Starting Training on {device}...")
pbar = tqdm(range(start_episode, max_episodes + 1), desc="Training")
for episode in pbar:
    state, _ = env.reset()
    current_ep_reward = 0

    # Use env.unwrapped.model.horizon to get the horizon from the base environment
    for t in range(env.unwrapped.model.horizon):
        time_step += 1
        action = ppo.select_action(state, memory)
        state, reward, done, truncated, _ = env.step(action)

        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        current_ep_reward += reward

        if time_step % update_timestep == 0:
            ppo.update(memory)
            memory.clear()
            time_step = 0

        if done or truncated:
            break

    running_reward += current_ep_reward
    logger.log(episode, current_ep_reward)

    if current_ep_reward > best_reward:
        best_reward = current_ep_reward
        ppo.save_checkpoint(best_model_path, episode, best_reward, time_step)

    if episode % save_interval == 0:
        ppo.save_checkpoint(latest_model_path, episode, best_reward, time_step)

    if episode % log_interval == 0:
        avg_reward = running_reward / log_interval
        pbar.set_postfix(
            {"Avg Rew": f"{avg_reward:.2f}", "Best": f"{best_reward:.2f}"}
        )
        running_reward = 0
        logger.plot()

env.close()
logger.plot() # Final plot