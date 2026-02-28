import os
import matplotlib

# Use a non-interactive backend for Matplotlib to avoid issues in headless environments (e.g., servers)
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    """
    Handles real-time logging of training metrics to TensorBoard.
    
    This includes:
    - Scalar metrics (Reward, Actor/Critic/Entropy Loss, KL Divergence, Clip Fraction).
    - Histograms of model parameters and gradients to monitor learning stability and vanishing/exploding gradients.
    """
    def __init__(self, log_dir):
        # The SummaryWriter is the primary entry point for logging data to TensorBoard
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Logs a single numerical value for a given step (e.g., epoch or episode)."""
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        """Logs a distribution of values (e.g., weight distributions) for a given step."""
        self.writer.add_histogram(tag, values, step)

    def log_model_stats(self, model, step):
        """
        Logs histograms of weights and gradient norms for all trainable parameters in the model.
        Useful for debugging training dynamics.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f"params/{name}", param.data.cpu().numpy(), step)
                if param.grad is not None:
                    # Log gradient norm to detect instability
                    self.log_scalar(f"gradients_norm/{name}", param.grad.norm().item(), step)
        self.writer.flush()

    def close(self):
        """Ensures all pending logs are written to disk before closing the writer."""
        self.writer.flush()
        self.writer.close()


class RewardLogger:
    """
    Manages persistent logging of episode rewards to a CSV file and generates summary plots.
    
    This class supports:
    - Appending rewards to an existing log for resumed training sessions.
    - Generating a static PNG plot with raw data and a moving average.
    """
    def __init__(self, log_dir, log_filename="rewards.csv"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_filename)
        self.episodes = []
        self.rewards = []

        # Ensure the output directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Resume logic: Load previous logs if the file already exists
        if os.path.exists(self.log_path):
            self.load_log()

    def load_log(self):
        """Loads historical reward data from the CSV file."""
        try:
            data = np.genfromtxt(self.log_path, delimiter=",", skip_header=1)
            if data.ndim == 1 and len(data) > 0:  # Handle case with only one logged episode
                self.episodes = [int(data[0])]
                self.rewards = [data[1]]
            elif data.ndim == 2:
                self.episodes = data[:, 0].astype(int).tolist()
                self.rewards = data[:, 1].tolist()
        except Exception as e:
            print(f"Error loading log: {e}")

    def clear_log(self):
        """Removes the existing log file and clears in-memory lists (used for fresh starts)."""
        self.episodes = []
        self.rewards = []
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def log(self, episode, reward):
        """Logs a new episode reward both in memory and by appending to the CSV file."""
        self.episodes.append(episode)
        self.rewards.append(reward)

        file_exists = os.path.exists(self.log_path)
        with open(self.log_path, "a") as f:
            if not file_exists:
                f.write("episode,reward\n")
            f.write(f"{episode},{reward}\n")

    def plot(self, save_path="reward_plot.png"):
        """Generates and saves a training progress plot using Matplotlib."""
        if not self.episodes:
            return

        plt.figure(figsize=(12, 6))
        
        # Plot raw episode rewards with high transparency
        plt.plot(self.episodes, self.rewards, label="Episode Reward", alpha=0.4, color="tab:blue")

        # Plot moving average for smoother visualization of trends
        if len(self.rewards) >= 10:
            window = 10
            moving_avg = np.convolve(self.rewards, np.ones(window) / window, mode="valid")
            plt.plot(
                self.episodes[window - 1 :],
                moving_avg,
                label=f"{window}-Episode Moving Average",
                color="tab:red",
                linewidth=2.5,
            )

        # Highlight the best reward achieved so far
        best_achieved = max(self.rewards)
        plt.axhline(y=best_achieved, color="tab:green", linestyle="--", label=f"Best Achievement: {best_achieved:.2f}")

        plt.xlabel("Episode Number")
        plt.ylabel("Cumulative Reward")
        plt.title("Training Progress: Reward over Episodes")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)

        # Save the figure to the specified output directory
        full_save_path = os.path.join(self.log_dir, save_path)
        plt.savefig(full_save_path, dpi=150)
        plt.close()
