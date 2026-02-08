# 🚂 Train Scheduler RL Project

This project uses Reinforcement Learning (PPO) to optimize train dwell times in a simulated rail system using **pyRDDLGym**.

## 📁 Project Structure
- `src/`: Core Python logic.
  - `train.py`: Main training loop.
  - `agent.py`: PPO implementation.
  - `wrappers.py`: Environment wrappers for skipping and vectorization.
  - `generator.py`: RDDL instance generator.
  - `visualizer.py`: Custom Matplotlib visualizer.
- `rddl/`: RDDL domain and instance files.
- `checkpoints/`: Saved model weights (git-ignored).
- `output/`: Visualization outputs and gifs (git-ignored).

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone git@github.com:beneldokow/Train_Scheduler_RL_proj.git
cd Train_Scheduler_RL_proj
```

### 2. Set up a Virtual Environment (Recommended)
This avoids conflicts with system-level Python packages.

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Training
```bash
python3 src/train.py
```

## 🧠 Domain Logic
The simulation runs on a custom RDDL domain where trains move between stations. The agent sets a `wait` time (0-10 minutes) at each station to minimize schedule deviation.

- **Observation Space:** Flattened vector of timers, train states, and passenger counts.
- **Action Space:** Discrete (0-10) minutes of additional dwell time.
- **Reward:** Negative absolute difference between planned and actual departure times.
