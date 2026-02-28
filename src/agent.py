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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        # Old policy used to calculate the probability ratio r(theta)
        self.policy_old = ActorCritic(state_dim, action_dim, 64).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.update_count = 0

    def select_action(self, state, memory):
        """Uses the old policy for interaction with the environment (standard PPO practice)."""
        return self.policy_old.act(state, memory)

    def update(self, memory, logger=None):
        """Perform a PPO update using the collected transitions."""
        
        # 1. Monte Carlo estimate of discounted rewards (returns)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards for stability
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert list of tensors in memory to single tensors
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # 2. Optimization Loop (K_epochs)
        for epoch in range(self.K_epochs):
            # Evaluating old actions and states with the current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Probability Ratio: r(theta) = pi_theta(a|s) / pi_theta_old(a|s)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Advantage Estimation: A(s,a) = G - V(s)
            advantages = rewards - state_values.detach()
            
            # Clipped Surrogate Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Total Loss = -min(surr1, surr2) + ValueLoss - EntropyBonus
            # Negative sign on min because we want to maximize the objective
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy # Encourages exploration

            loss = actor_loss + critic_loss + entropy_loss

            # 3. Gradient Descent
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            # Logging (if logger provided)
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

        # Update the old policy weights
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
