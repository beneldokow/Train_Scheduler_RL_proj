# Project Context: Train Scheduling RL

## Goal
Optimize train dwell times (0-10 mins) at stations to minimize schedule deviation using PPO Reinforcement Learning. The environment is simulated using pyRDDLGym.

## Architecture
The project is modularized into `src/`.

1. **Domain (`rddl/domain.rddl`)**: Defines the physics.
   - States: `train_timer`, `passengers_at_station`, `current_state`.
   - Actions: `wait(station)` (int 0-10).
   - Reward: Linear penalty for lateness (Planned - Actual).

2. **Wrappers (`src/wrappers.py`)**:
   - `RDDLDecisionWrapper`: Skips empty simulation steps until a train arrives at a station. Handles logic for "active" stations.
   - `PPOAdapter`: Flattens the dictionary state into a vector for PyTorch.

3. **Agent (`src/agent.py`)**:
   - Implements PPO with Actor-Critic networks in PyTorch.
   - Includes Memory buffer for experience replay.

4. **Visualizer (`src/visualizer.py`)**:
   - Custom Matplotlib rendering of circular train tracks.

## Current Status
- The environment works.
- PPO agent is implemented but needs hyperparameter tuning.
- Instance generation is automated in `src/generator.py`.

## Rules for Changes
- Always output the full file content when suggesting code changes.
- Maintain the import structure (`from src.xxx import yyy`).