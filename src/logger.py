import os
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
    Manages persistent logging of episode rewards to a CSV file.
    """
    def __init__(self, log_dir, log_filename="rewards.csv"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_filename)

        # Ensure the output directory exists
        os.makedirs(self.log_dir, exist_ok=True)

    def clear_log(self):
        """Removes the existing log file (used for fresh starts)."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def log(self, episode, reward):
        """Logs a new episode reward by appending to the CSV file."""
        file_exists = os.path.exists(self.log_path)
        with open(self.log_path, "a") as f:
            if not file_exists:
                f.write("episode,reward\n")
            f.write(f"{episode},{reward}\n")
