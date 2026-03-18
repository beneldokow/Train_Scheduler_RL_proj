# 🚂 Train Scheduler RL Project

This project uses Reinforcement Learning (PPO) to optimize train dwell times in a simulated rail system using **pyRDDLGym**.

## 🎯 Review Quick Summary (The "Elevator Pitch")
This project tackles the **Train Scheduling Optimization** problem using **Deep Reinforcement Learning (PPO)**. 
- **Problem:** Dynamic passenger arrivals cause delays that ripple through the network.
- **Goal:** Minimize station wait times and departure delays across a multi-station, multi-train system.
- **Innovation:** Uses **RDDL (Relational Dynamic Bayesian Networks)** for robust environment modeling and a custom **PPO (Proximal Policy Optimization)** agent for decision-making.

## 🧠 Algorithm Technicals (Deep Dive)
The core learning logic is implemented in `src/agent.py` using **PPO with Clipped Objective**:
- **Policy Network:** An Actor-Critic architecture that outputs discrete dwell-time actions.
- **Optimization:** Uses **Monte Carlo returns** for advantage estimation, providing an unbiased signal for policy updates.
- **Robustness:** Includes **Entropy Regularization** to encourage exploration and prevent premature convergence.
- **Normalization:** Observations are pre-processed through a normalization layer to ensure stable training across different station/train counts.

## 📁 Project Structure
- `src/`: Core Python logic.
  - `train.py`: Main training loop.
  - `agent.py`: PPO implementation.
  - `wrappers.py`: Environment wrappers for skipping and vectorization.
  - `generator.py`: RDDL instance generator.
  - `visualizer.py`: Custom Matplotlib visualizer for episode GIFs.
  - `logger.py`: TensorBoard and CSV reward logging utilities.
- `rddl/`: RDDL domain and instance files.
- `checkpoints/`: Saved model weights (git-ignored).
- `output/`: Visualization outputs, gifs, and TensorBoard logs (git-ignored).

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone git@github.com:beneldokow/Train_Scheduler_RL_proj.git
cd Train_Scheduler_RL_proj
```

### 2. Run Training (Automated Setup)
The project includes a `run.sh` script that handles virtual environment creation and dependency management automatically, even on restricted filesystems like Google Drive.

**Basic Run:**
```bash
./run.sh
```
*Note: The first run will prompt you to create a virtual environment if one is not detected.*

**Run with Custom Arguments:**
```bash
# Run for 1000 episodes, log every 10 episodes, and start fresh (ignore checkpoints)
./run.sh --episodes 1000 --log_interval 10 --force_restart
```

### 3. Monitoring & Visualization

#### TensorBoard (Primary Dashboard)
Real-time metrics, including episode rewards, actor/critic losses, and parameter distributions, are logged to TensorBoard. The `run.sh` script automatically launches the dashboard after training.

**Manual Launch:**
```bash
tensorboard --logdir output/tensorboard
```

#### Episode Visualizations
GIFs of agent behavior are saved to `output/visualizations/` periodically, showing real-time train movement and station crowding.

---

## 🛠 Available Arguments
- `--episodes`: Maximum number of episodes (target) (default: 5000).
- `--additional_episodes`: Number of additional episodes to run from current checkpoint.
- `--log_interval`: How often to update the log (default: 20).
- `--save_interval`: How often to save checkpoints (default: 50).
- `--force_restart`: Ignore existing checkpoints and start training from scratch.
- `--reuse [<name>]`: (run.sh only) Use a pre-existing instance from `rddl/instances/`.
- `--instance_path <path>`: Run with a specific RDDL instance file path.
- `--num_trains`: Number of trains for generated instance (default: 3).
- `--num_stations`: Number of stations for generated instance (default: 4).
- `--variance_factor`: Controls randomness of passenger arrivals (default: 0.2).

## 🧠 Domain Logic & Features
- **Stochastic Arrivals:** Passenger arrivals follow a Normal distribution that scales with time-steps for consistency.
- **Auto-Horizon:** Optimized simulation length calculated as `(2 * trains * stations) + 10`.
- **Instance-Aware Checkpoints:** Models are saved per-instance and variance level (e.g., `latest_model_small_3s_2t_v20.pth`).
- **Parameter Tracking:** If `num_trains`, `num_stations`, or `variance_factor` change, the system automatically triggers a fresh start to avoid dimension mismatches.
- **Robust Venv Management:** `run.sh` saves the path to your functional venv in `.venv_path`, allowing it to reside outside of Google Drive for better performance and reliability.

### 📊 Agent Interface
- **Observation Space:** Flattened vector of timers, train states, and passenger counts.
- **Action Space:** Discrete (0-10) minutes of additional dwell time.
- **Reward:** Negative absolute difference between planned and actual departure times.
