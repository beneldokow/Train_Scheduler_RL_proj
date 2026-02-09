import os
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_model_stats(self, model, step):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f"params/{name}", param.data.cpu().numpy(), step)
                if param.grad is not None:
                    self.log_scalar(
                        f"gradients_norm/{name}", param.grad.norm().item(), step
                    )
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()


class RewardLogger:
    def __init__(self, log_dir, log_filename="rewards.csv"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_filename)
        self.episodes = []
        self.rewards = []

        os.makedirs(self.log_dir, exist_ok=True)

        # If log file exists, load it
        if os.path.exists(self.log_path):
            self.load_log()

    def load_log(self):
        try:
            data = np.genfromtxt(self.log_path, delimiter=",", skip_header=1)
            if data.ndim == 1 and len(data) > 0:  # Handle single row
                self.episodes = [int(data[0])]
                self.rewards = [data[1]]
            elif data.ndim == 2:
                self.episodes = data[:, 0].astype(int).tolist()
                self.rewards = data[:, 1].tolist()
        except Exception as e:
            print(f"Error loading log: {e}")

    def clear_log(self):
        self.episodes = []
        self.rewards = []
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def log(self, episode, reward):
        self.episodes.append(episode)
        self.rewards.append(reward)

        # Append to file
        file_exists = os.path.exists(self.log_path)
        with open(self.log_path, "a") as f:
            if not file_exists:
                f.write("episode,reward\n")
            f.write(f"{episode},{reward}\n")

    def plot(self, save_path="reward_plot.png"):
        if not self.episodes:
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.episodes, self.rewards, label="Episode Reward", alpha=0.6)

        # Add moving average if enough data
        if len(self.rewards) >= 10:
            window = 10
            moving_avg = np.convolve(
                self.rewards, np.ones(window) / window, mode="valid"
            )
            plt.plot(
                self.episodes[window - 1 :],
                moving_avg,
                label=f"{window}-ep Moving Avg",
                color="red",
                linewidth=2,
            )

        # Add a horizontal line for the best reward
        best_achieved = max(self.rewards)
        plt.axhline(
            y=best_achieved,
            color="green",
            linestyle="--",
            label=f"Best: {best_achieved:.2f}",
        )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.legend()
        plt.grid(True, alpha=0.3)

        full_save_path = os.path.join(self.log_dir, save_path)
        plt.savefig(full_save_path)
        plt.close()
