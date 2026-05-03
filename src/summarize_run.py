import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def generate_summary(run_name):
    base_path = os.path.join("history", run_name)
    log_dir = os.path.join(base_path, "output", "tensorboard")
    output_csv = os.path.join(base_path, "output", "reward_summary.csv")

    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} not found.")
        return

    print(f"Processing TensorBoard logs from {log_dir}...")
    acc = EventAccumulator(log_dir)
    acc.Reload()

    # Extract all available scalar tags
    tags = acc.Tags().get("scalars", [])
    
    # We will merge all data into one DataFrame
    all_data = {}
    
    for tag in tags:
        events = acc.Scalars(tag)
        # Use the tag name as the column name (sanitized)
        col_name = tag.replace("/", "_")
        all_data[col_name] = pd.DataFrame(
            [(e.step, e.value) for e in events], columns=["episode", col_name]
        ).set_index("episode")

    if not all_data:
        print("No scalar data found.")
        return

    # Merge all metrics on episode index
    final_df = pd.concat(all_data.values(), axis=1)
    
    final_df.to_csv(output_csv)
    print(f"Summary saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="Name of the run in history/")
    args = parser.parse_args()
    generate_summary(args.run_name)
