# 🚂 Train Scheduler RL

Optimizing railway dwell times using **Deep Reinforcement Learning (PPO)**. This repository provides a complete pipeline to simulate, train, and benchmark intelligent agents on stochastic rail networks.

## 🎯 Overview
This project addresses the **Train Scheduling Optimization** problem, where dynamic passenger arrivals cause delays that ripple through the network.
- **The Problem:** In a stochastic system, fixed schedules fail when passenger boarding and disembarking times vary, leading to network-wide congestion.
- **The Goal:** Minimize station wait times and departure delays across multi-station, multi-train systems by learning optimal adaptive dwell-time policies.
- **The Innovation:** Uses **RDDL (Relational Dynamic Bayesian Networks)** for robust environment modeling and a custom **PPO (Proximal Policy Optimization)** agent for decision-making at scale.

## 📁 Repository Structure
*   **`src/`**: Core logic and simulation infrastructure.
    *   `train.py`: Training orchestrator (checkpointing, logging).
    *   `agent.py`: PPO implementation (Actor-Critic architecture).
    *   `generator.py`: Instance creator (generates valid RDDL environments with embedded schedules).
    *   `logger.py`: TensorBoard metrics & CSV reward persistence.
    *   `summarize_run.py`: Aggregator for converting TB logs to compact CSV summaries for Git versioning.
    *   `visualizer.py`: Generates episode playback GIFs.
    *   `wrappers.py`: Gym-compatible environment adapters.
    *   **`config/`**: Contains `requirements.txt` for environment setup.
*   **`rddl/`**: Environment files.
    *   `domain.rddl`: Rail network physics and rules.
    *   `instances/`: Reusable RDDL instance files.
*   **`experiments/`**: Git-tracked archive of significant training runs (configs, checkpoints, CSV summaries).
*   **`output/` & `checkpoints/`**: Git-ignored scratchpad folders for temporary training files.
*   **`run.sh`**: Unified entry point for environment setup and training.

## 🚀 Quick Start
The `run.sh` script is the **unified entry point**. It handles environment creation, dependency management, and training execution automatically.

### Setup & Train
```bash
git clone git@github.com:beneldokow/Train_Scheduler_RL_proj.git
cd Train_Scheduler_RL_proj
./run.sh --episodes 5000 --run_name my_benchmark
```
*Note: The script creates a `.venv_path` file to ensure the environment persists between runs.*

## 🛠 Training Options & Flags
The `run.sh` script passes these flags directly to `src/train.py`:

| Flag | Description |
| :--- | :--- |
| `--episodes` | Total training episodes (default: 5000). |
| `--run_name` | Saves logs/checkpoints to `experiments/<run_name>/` (Git-tracked). |
| `--instance_path` | Path to a specific RDDL instance file. |
| `--num_trains` | (Generated instances) Number of trains for the simulation. |
| `--num_stations` | (Generated instances) Number of stations for the simulation. |
| `--force_restart` | Discards checkpoints and starts training from scratch. |
| `--save_interval` | Frequency of checkpoint creation (default: 50). |

## 🧪 Operational Modes
1. **Generated Run:** Use `--num_trains` and `--num_stations`. The `generator.py` script builds a physically consistent network and computes the target schedule automatically. **Always prefer this for new experiments.**
2. **Existing Instance:** Use `--instance_path rddl/instances/<name>.rddl` to run on a specific, previously generated instance.
3. **Resuming:** If a run stops, execute the same command again; the system detects existing checkpoints in the `checkpoints/` folder and resumes training.

> **⚠️ Conflict Warning:** If you pass both generation flags and an `--instance_path`, the script ignores the generation flags and prioritizes the instance file.

## 📊 Monitoring & Versioning
*   **TensorBoard:** Real-time metrics are logged to `output/tensorboard/`.
    *   *Launch:* `./run.sh` automatically launches TensorBoard upon training completion. Alternatively: `tensorboard --logdir output/tensorboard`.
*   **Storage Limits:** TensorBoard event files are often too large (>100MB) for GitHub. **Do not commit these to Git.**
*   **Experiment History:** Use `--run_name` to move significant results to `experiments/`. Use `python3 src/summarize_run.py <run_name>` to generate a tiny `reward_summary.csv` for Git tracking.

---
*Built with PyTorch, pyRDDLGym, and TensorBoard.*
