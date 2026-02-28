import os
import time
import torch
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

"""
Interactive Dashboard Generator for Train Scheduler RL.

This module creates a standalone HTML dashboard using Plotly.
It combines:
1. Training Metrics (from TensorBoard logs): Rewards, Actor/Critic losses, KL Divergence.
2. Weight Distributions (from .pth checkpoints): Histograms of neural network parameters.
3. Gradient Norms: To monitor learning stability and identify potential vanishing/exploding gradients.

The dashboard is designed for deep-dive analysis of agent convergence and network health.
"""

def extract_tensorboard_data(log_dir):
    """
    Parses TensorBoard event files and converts scalar metrics to Pandas DataFrames.
    
    Args:
        log_dir: Directory containing TensorBoard tfevents files.
    Returns:
        A dictionary mapping metric tags (e.g., 'reward/episode') to DataFrames.
    """
    # Wait a moment to ensure file handles are released if called immediately after training
    time.sleep(1)
    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()

    tags = acc.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = acc.Scalars(tag)
        data[tag] = pd.DataFrame(
            [(e.step, e.value) for e in events], columns=["step", "value"]
        )
    return data


def create_dashboard(log_dir, model_path, output_html):
    """
    Main entry point for dashboard generation.
    
    1. Extracts scalar logs and weight checkpoints.
    2. Organizes metrics into themed groups (Core RL, Gradients, Weights).
    3. Builds a dynamic Plotly grid based on the number of available metrics.
    4. Saves the result as an interactive HTML file.
    """
    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} does not exist.")
        return

    # 1. Data Extraction
    data = extract_tensorboard_data(log_dir)

    # 2. Weight Extraction: Load the model checkpoint to visualize parameter distributions
    weights_data = {}
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            # Handle cases where the checkpoint is either the raw state_dict or a metadata dict
            model_state = checkpoint.get("model_state", checkpoint)
            for name, param in model_state.items():
                weights_data[name] = param.numpy().flatten()
        except Exception as e:
            print(f"Warning: Could not load weights for histograms: {e}")

    if not data and not weights_data:
        print("No data found to visualize.")
        return

    # 3. Organization: Group related metrics for a cleaner UI
    groups = {
        "Core RL Metrics": [
            "reward/episode",
            "loss/total",
            "loss/actor",
            "loss/critic",
            "loss/entropy",
            "stats/approx_kl",
            "stats/clip_fraction",
        ],
        "Gradient Norms": sorted([t for t in data.keys() if t.startswith("gradients_norm/")]),
    }

    scalar_tags = []
    for g_name in ["Core RL Metrics", "Gradient Norms"]:
        for t in groups.get(g_name, []):
            if t in data: scalar_tags.append((t, g_name))

    # Add any remaining scalars not in the predefined groups
    known_tags = [t for v in groups.values() for t in v]
    for t in sorted(data.keys()):
        if t not in known_tags: scalar_tags.append((t, "Other Metrics"))

    # 4. Grid Calculation
    cols = 3
    num_scalars = len(scalar_tags)
    num_histograms = len(weights_data)
    scalar_rows = (num_scalars + cols - 1) // cols
    hist_rows = (num_histograms + cols - 1) // cols
    total_rows = scalar_rows + hist_rows

    # Build Plot Titles with padding to ensure correct alignment in the subplot grid
    titles = []
    for t, g in scalar_tags: titles.append(f"[{g}] {t.split('/')[-1]}")
    padding_needed = (cols - (num_scalars % cols)) % cols
    for _ in range(padding_needed): titles.append("")
    for name in weights_data.keys(): titles.append(f"[Weight Dist] {name}")

    # 5. Dashboard Construction
    fig = make_subplots(
        rows=total_rows, cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.07,
    )

    # Add Scalar Line Charts
    for i, (tag, group) in enumerate(scalar_tags):
        row, col = (i // cols) + 1, (i % cols) + 1
        df = data[tag]
        fig.add_trace(go.Scatter(x=df["step"], y=df["value"], name=tag, mode="lines"), row=row, col=col)
        fig.update_xaxes(title_text="Step", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)

    # Add Weight Histograms
    for i, (name, values) in enumerate(weights_data.items()):
        row, col = scalar_rows + (i // cols) + 1, (i % cols) + 1
        fig.add_trace(go.Histogram(x=values, name=name, nbinsx=30, marker_color="rgba(0, 0, 255, 0.6)"), row=row, col=col)
        fig.update_xaxes(title_text="Param Value", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    # Final Styling
    total_height = max(800, 400 * total_rows)
    fig.update_layout(
        height=total_height, width=1600,
        title_text="Train Scheduler RL - Comprehensive Analysis Dashboard",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # 6. Save and Report
    fig.write_html(output_html)
    print(f"Dashboard generated: {num_scalars} metrics and {num_histograms} parameter distributions saved to {output_html}")


if __name__ == "__main__":
    # Convenience execution: tries to generate a dashboard from default paths
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_DIR = os.path.join(BASE_PATH, "output", "tensorboard")
    # Note: latest_model.pth is a generic fallback; train.py uses instance-specific names
    MODEL_PATH = os.path.join(BASE_PATH, "checkpoints", "latest_model.pth")
    OUTPUT_FILE = os.path.join(BASE_PATH, "output", "training_dashboard.html")
    create_dashboard(LOG_DIR, MODEL_PATH, OUTPUT_FILE)
