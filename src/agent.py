import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = (
            [],
            [],
            [],
            [],
            [],
        )

    def clear(self):
        del (
            self.actions[:],
            self.states[:],
            self.logprobs[:],
            self.rewards[:],
            self.is_terminals[:],
        )


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1),
        )

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip):
        self.lr, self.betas, self.gamma, self.eps_clip, self.K_epochs = (
            lr,
            betas,
            gamma,
            eps_clip,
            K_epochs,
        )
        self.policy = ActorCritic(state_dim, action_dim, 64).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, 64).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.update_count = 0

    def select_action(self, state, memory):
        return self.policy_old.act(state, memory)

    def update(self, memory, logger=None):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        for epoch in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            actor_loss = -torch.min(surr1, surr2)
            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy

            loss = actor_loss + critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.mean().backward()

            if logger and epoch == self.K_epochs - 1:
                # Log metrics for the last epoch of the update
                with torch.no_grad():
                    approx_kl = (old_logprobs - logprobs).mean()
                    clip_frac = (torch.abs(ratios - 1) > self.eps_clip).float().mean()

                logger.log_scalar("loss/total", loss.mean().item(), self.update_count)
                logger.log_scalar(
                    "loss/actor", actor_loss.mean().item(), self.update_count
                )
                logger.log_scalar(
                    "loss/critic", critic_loss.mean().item(), self.update_count
                )
                logger.log_scalar(
                    "loss/entropy", dist_entropy.mean().item(), self.update_count
                )
                logger.log_scalar(
                    "stats/approx_kl", approx_kl.item(), self.update_count
                )
                logger.log_scalar(
                    "stats/clip_fraction", clip_frac.item(), self.update_count
                )
                logger.log_model_stats(self.policy, self.update_count)

            self.optimizer.step()

        self.update_count += 1
        self.policy_old.load_state_dict(self.policy.state_dict())

    # --- UPDATED SAVE FUNCTION ---
    def save_checkpoint(self, path, episode_num, best_reward, time_step):
        torch.save(
            {
                "model_state": self.policy_old.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "episode": episode_num,
                "best_reward": best_reward,
                "time_step": time_step,
                "update_count": self.update_count,
            },
            path,
        )

    # --- UPDATED LOAD FUNCTION ---
    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=device)

        # Load weights
        self.policy.load_state_dict(checkpoint["model_state"])
        self.policy_old.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update_count = checkpoint.get("update_count", 0)

        # Return the saved metadata
        return (
            checkpoint.get("episode", 1),
            checkpoint.get("best_reward", -float("inf")),
            checkpoint.get("time_step", 0),
        )
