import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

# Device configuration: Use GPU if available for faster training.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    """
    Buffer to store episode transitions for PPO updates.
    Stores: states, actions, log probabilities of actions, rewards, and terminal flags.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """Clears the buffer after an update."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    """
    Shared Actor-Critic Network Architecture.
    
    Actor (Policy): Outputs a probability distribution over discrete actions.
    Critic (Value): Estimates the state-value function V(s).
    """
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        
        # Actor Network: Predicts action probabilities
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1),
        )
        
        # Critic Network: Predicts the expected return (value) of the state
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1),
        )

    def act(self, state, memory):
        """Selects an action based on the current policy and stores state/action in memory."""
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()

    def evaluate(self, state, action):
        """Evaluates actions against the current policy for PPO updates."""
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    """
    Proximal Policy Optimization (PPO) Agent.
    
    Uses a clipped objective function to ensure stable policy updates by preventing 
    large updates that could collapse performance.
    """
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma          # Discount factor for rewards
        self.eps_clip = eps_clip    # Clipping parameter for the ratio r(theta)
        self.K_epochs = K_epochs    # Number of optimization epochs per update
        
        # Current policy being optimized
        self.policy = ActorCritic(state_dim, action_dim, 64).to(device)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr, alpha=0.99, eps=1e-5)
        
        # Old policy used to calculate the probability ratio r(theta)
        self.policy_old = ActorCritic(state_dim, action_dim, 64).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.update_count = 0

    def select_action(self, state, memory):
        """Uses the old policy for interaction with the environment (standard PPO practice)."""
        return self.policy_old.act(state, memory)

    def update(self, memory, logger=None):
        """
        PPO Update: Optimizes the Actor-Critic networks using collected experience.
        The core of PPO: Clipped surrogate objective + Value Loss + Entropy Bonus.
        """
        
        # 1. Compute 'Returns': The discounted sum of future rewards (Monte Carlo)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing rewards stabilizes training by keeping gradients at a consistent scale
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert lists to tensors for batch processing on GPU/CPU
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # 2. Optimization Loop: PPO reuses the same data for K epochs for efficiency
        for epoch in range(self.K_epochs):
            # Evaluate the current batch under the *new* (changing) policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Probability Ratio: How much the policy has changed since data collection
            # ratio = exp(new_logprob - old_logprob)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Advantage Estimation: How much better was this action than the Critic's baseline?
            advantages = rewards - state_values.detach()
            
            # Clipped Surrogate Objective: The heart of PPO.
            # Limits the 'incentive' to change the policy too much in one update.
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Total Loss: Combine Policy improvement, Value accuracy, and Exploration
            actor_loss = -torch.min(surr1, surr2) # Negative because we maximize reward
            critic_loss = 0.5 * self.MseLoss(state_values, rewards) # Minimize error in V(s)
            entropy_loss = -0.01 * dist_entropy # Bonus for 'uncertainty' to keep exploring

            loss = actor_loss + critic_loss + entropy_loss

            # 3. Gradient Descent
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Periodic logging of internal metrics (KL divergence, clip fraction)
            if logger and epoch == self.K_epochs - 1:
                with torch.no_grad():
                    approx_kl = (old_logprobs - logprobs).mean()
                    clip_frac = (torch.abs(ratios - 1) > self.eps_clip).float().mean()
                
                logger.log_scalar("loss/total", loss.mean().item(), self.update_count)
                logger.log_scalar("loss/actor", actor_loss.mean().item(), self.update_count)
                logger.log_scalar("loss/critic", critic_loss.mean().item(), self.update_count)
                logger.log_scalar("loss/entropy", dist_entropy.mean().item(), self.update_count)
                logger.log_scalar("stats/approx_kl", approx_kl.item(), self.update_count)
                logger.log_scalar("stats/clip_fraction", clip_frac.item(), self.update_count)
                logger.log_model_stats(self.policy, self.update_count)

            self.optimizer.step()

        # Finalize Update: Synchronize the 'old' policy with the newly optimized one
        self.update_count += 1
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_checkpoint(self, path, episode_num, best_reward, time_step):
        """Saves model weights and training metadata."""
        torch.save({
            "model_state": self.policy_old.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "episode": episode_num,
            "best_reward": best_reward,
            "time_step": time_step,
            "update_count": self.update_count,
        }, path)

    def load_checkpoint(self, path):
        """Loads model weights and training metadata from a checkpoint file."""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=device)

        self.policy.load_state_dict(checkpoint["model_state"])
        self.policy_old.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update_count = checkpoint.get("update_count", 0)

        return (
            checkpoint.get("episode", 1),
            checkpoint.get("best_reward", -float("inf")),
            checkpoint.get("time_step", 0),
        )
