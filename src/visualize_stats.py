import os
import time
import torch
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tensorboard_data(log_dir):
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
    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} does not exist.")
        return

    # 1. Extract Scalar Data
    data = extract_tensorboard_data(log_dir)

    # 2. Load Model Weights for Histograms
    weights_data = {}
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model_state = checkpoint.get("model_state", checkpoint)
            for name, param in model_state.items():
                weights_data[name] = param.numpy().flatten()
        except Exception as e:
            print(f"Warning: Could not load weights for histograms: {e}")

    if not data and not weights_data:
        print("No data found to visualize.")
        return

    # Define groups for organized visualization
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
        "Gradient Norms": sorted(
            [t for t in data.keys() if t.startswith("gradients_norm/")]
        ),
    }

    scalar_tags = []
    for g_name in ["Core RL Metrics", "Gradient Norms"]:
        for t in groups.get(g_name, []):
            if t in data:
                scalar_tags.append((t, g_name))

    # Other scalars
    known_tags = [t for v in groups.values() for t in v]
    for t in sorted(data.keys()):
        if t not in known_tags:
            scalar_tags.append((t, "Other Metrics"))

    # Calculate Grid
    cols = 3
    num_scalars = len(scalar_tags)
    num_histograms = len(weights_data)

    scalar_rows = (num_scalars + cols - 1) // cols
    hist_rows = (num_histograms + cols - 1) // cols
    total_rows = scalar_rows + hist_rows

    # FIXED: Build titles list with padding to align with grid positions
    titles = []
    # Add scalar titles
    for t, g in scalar_tags:
        titles.append(f"[{g}] {t.split('/')[-1]}")

    # Add padding to finish the last row of scalars
    padding_needed = (cols - (num_scalars % cols)) % cols
    for _ in range(padding_needed):
        titles.append("")

    # Add histogram titles
    for name in weights_data.keys():
        titles.append(f"[Distribution] {name}")

    fig = make_subplots(
        rows=total_rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.07,
    )

    # Plot Scalars
    for i, (tag, group) in enumerate(scalar_tags):
        row = (i // cols) + 1
        col = (i % cols) + 1
        df = data[tag]
        fig.add_trace(
            go.Scatter(x=df["step"], y=df["value"], name=tag, mode="lines"),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Step", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)

    # Plot Histograms
    for i, (name, values) in enumerate(weights_data.items()):
        # Histograms start on a fresh row after all scalar rows
        row = scalar_rows + (i // cols) + 1
        col = (i % cols) + 1
        fig.add_trace(
            go.Histogram(
                x=values, name=name, nbinsx=30, marker_color="rgba(0, 0, 255, 0.6)"
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Weight/Bias Value", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    total_height = max(800, 400 * total_rows)
    fig.update_layout(
        height=total_height,
        width=1600,
        title_text="Train Scheduler RL - Comprehensive Debugging Dashboard",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    fig.write_html(output_html)
    print(
        f"Corrected dashboard with {num_scalars} scalars and {num_histograms} histograms saved."
    )


if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_DIR = os.path.join(BASE_PATH, "output", "tensorboard")
    MODEL_PATH = os.path.join(BASE_PATH, "checkpoints", "latest_model.pth")
    OUTPUT_FILE = os.path.join(BASE_PATH, "output", "training_dashboard.html")
    create_dashboard(LOG_DIR, MODEL_PATH, OUTPUT_FILE)
