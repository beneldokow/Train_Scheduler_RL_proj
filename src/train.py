import os
import sys
import torch
import argparse
import pyRDDLGym
import warnings
import json
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
from src.logger import RewardLogger, TensorboardLogger
from src.visualize_stats import create_dashboard

# Argument Parser
parser = argparse.ArgumentParser(description="Train the Scheduler RL Agent")
parser.add_argument(
    "--episodes", type=int, default=5000, help="Maximum number of episodes (target)"
)
parser.add_argument(
    "--additional_episodes",
    type=int,
    default=None,
    help="Number of additional episodes to run from current checkpoint",
)
parser.add_argument(
    "--log_interval", type=int, default=20, help="Log interval for rewards"
)
parser.add_argument(
    "--save_interval", type=int, default=50, help="Checkpoint save interval"
)
parser.add_argument(
    "--update_timestep", type=int, default=2000, help="PPO update timestep"
)
parser.add_argument(
    "--force_restart", action="store_true", help="Ignore checkpoints and start fresh"
)
parser.add_argument(
    "--instance_path",
    type=str,
    default=None,
    help="Path to specific RDDL instance file",
)
parser.add_argument(
    "--num_trains", type=int, default=3, help="Number of trains for generated instance"
)
parser.add_argument(
    "--num_stations",
    type=int,
    default=4,
    help="Number of stations for generated instance",
)
parser.add_argument(
    "--variance_factor",
    type=float,
    default=0.2,
    help="Variance factor for passenger arrivals",
)
args = parser.parse_args()

# Configuration
# BASE_PATH is already defined above
RDDL_DIR = os.path.join(BASE_PATH, "rddl")
OUTPUT_DIR = os.path.join(BASE_PATH, "output")
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "tensorboard")
DOMAIN_PATH = os.path.join(RDDL_DIR, "domain.rddl")
DEFAULT_INSTANCE_PATH = os.path.join(RDDL_DIR, "instance.rddl")

checkpoint_dir = os.path.join(BASE_PATH, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Instance Path logic
if args.instance_path:
    INSTANCE_PATH = args.instance_path
    if not os.path.isabs(INSTANCE_PATH):
        INSTANCE_PATH = os.path.join(BASE_PATH, INSTANCE_PATH)
else:
    INSTANCE_PATH = DEFAULT_INSTANCE_PATH

# Logic to detect parameter changes for the default instance
config_tracker_path = os.path.join(checkpoint_dir, "instance_config.json")
current_config = {
    "num_trains": args.num_trains,
    "num_stations": args.num_stations,
    "variance_factor": args.variance_factor,
}

should_force_restart = args.force_restart

# Only track config for the default generated instance
if INSTANCE_PATH == DEFAULT_INSTANCE_PATH:
    if os.path.exists(config_tracker_path):
        with open(config_tracker_path, "r") as f:
            try:
                last_config = json.load(f)
                if last_config != current_config:
                    print(
                        "Detected parameter change (trains/stations/variance). Forcing fresh start..."
                    )
                    should_force_restart = True
            except json.JSONDecodeError:
                should_force_restart = True

# Initialize Loggers
logger = RewardLogger(OUTPUT_DIR)
tb_logger = TensorboardLogger(TENSORBOARD_DIR)

# 1. Generate Instance (only if missing or forced, and only if not using specific instance_path)
if (
    should_force_restart or not os.path.exists(INSTANCE_PATH)
) and not args.instance_path:
    print(
        f"Generating instance with {args.num_trains} trains and {args.num_stations} stations (Var Factor: {args.variance_factor})..."
    )
    rddl_content = generate_instance(
        num_trains=args.num_trains,
        num_stations=args.num_stations,
        variance_factor=args.variance_factor,
    )
    with open(INSTANCE_PATH, "w") as f:
        f.write(rddl_content)

    # Save the config we just generated
    if INSTANCE_PATH == DEFAULT_INSTANCE_PATH:
        with open(config_tracker_path, "w") as f:
            json.dump(current_config, f)
elif args.instance_path:
    print(f"Using provided instance: {INSTANCE_PATH}")
else:
    print(f"Using existing instance: {INSTANCE_PATH}")

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
# (Moved checkpoint_dir up)

# Derive instance name for checkpointing
instance_base = os.path.basename(INSTANCE_PATH).replace(".rddl", "")
# Include variance in the name to distinguish noise levels
instance_name = f"{instance_base}_v{int(args.variance_factor*100)}"

latest_model_path = os.path.join(checkpoint_dir, f"latest_model_{instance_name}.pth")
best_model_path = os.path.join(checkpoint_dir, f"best_model_{instance_name}.pth")

start_episode = 1
max_episodes = args.episodes
update_timestep = args.update_timestep
log_interval = args.log_interval
save_interval = args.save_interval
time_step = 0
running_reward = 0
best_reward = -float("inf")

# Resume logic: Check latest first, then best
resume_path = None
if not should_force_restart:
    if os.path.exists(latest_model_path):
        resume_path = latest_model_path
    elif os.path.exists(best_model_path):
        resume_path = best_model_path

if resume_path:
    start_episode, best_reward, time_step = ppo.load_checkpoint(resume_path)
    start_episode += 1
    print(
        f"Resuming from episode {start_episode} using {os.path.basename(resume_path)} (Best Reward: {best_reward:.2f})"
    )
else:
    logger.clear_log()
    # Clear old TensorBoard logs on fresh start
    if os.path.exists(TENSORBOARD_DIR):
        import shutil

        shutil.rmtree(TENSORBOARD_DIR)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

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
            ppo.update(memory, logger=tb_logger)
            memory.clear()
            time_step = 0

        if done or truncated:
            break

    running_reward += current_ep_reward
    logger.log(episode, current_ep_reward)
    tb_logger.log_scalar("reward/episode", current_ep_reward, episode)

    if current_ep_reward > best_reward:
        best_reward = current_ep_reward
        ppo.save_checkpoint(best_model_path, episode, best_reward, time_step)

    if episode % save_interval == 0:
        ppo.save_checkpoint(latest_model_path, episode, best_reward, time_step)

    if episode % log_interval == 0:
        avg_reward = running_reward / log_interval
        pbar.set_postfix({"Avg Rew": f"{avg_reward:.2f}", "Best": f"{best_reward:.2f}"})
        running_reward = 0
        logger.plot()

env.close()
logger.plot()  # Final plot
tb_logger.close()

# Generate interactive dashboard
DASHBOARD_PATH = os.path.join(OUTPUT_DIR, "training_dashboard.html")
create_dashboard(TENSORBOARD_DIR, latest_model_path, DASHBOARD_PATH)
