# 🚂 Deep Context: Train Scheduling RL Project

## 1. Project Overview
This project simulates a train system using **pyRDDLGym** and optimizes schedule adherence using a **PPO (Proximal Policy Optimization)** Reinforcement Learning agent. The system was recently refactored from a Colab Notebook into a modular Python package.

## 2. Core Physics (RDDL Domain)
The simulation runs on `rddl/domain.rddl`.
* **Time:** Continuous time, but discrete events. The simulation "jumps" (`global_timer`) to the next event (arrival or departure).
* **State:**
    * `train_timer(t)`: Time until this specific train reaches its next node.
    * `current_time`: **Absolute** simulation time (accumulates `global_timer` at every step).
    * `passengers_at_station(s)`: Linearly increases based on `PASSENGER_ARRIVAL_RATE`.
* **Actions:**
    * **Action Name:** `wait(station)` (Renamed from `wait_action` to avoid variable collisions).
    * **Range:** Integer `0` to `10` minutes.
    * **Logic:** When a train is `TRAIN_ACTIVE` (at station), the agent sets a `wait` time.
* **Reward Function (Critical):**
    * Goal: Minimize delay.
    * Formula: `PLANNED_DEPARTURE - ACTUAL_DEPARTURE`.
    * Optimization: This was optimized from $O(T \times S)$ to **Linear $O(T)$** by directly indexing: `PLANNED_DEPARTURE_TIME(?t, current_station(?t))`.

## 3. The Python Architecture (`src/`)

### A. The Environment Wrappers (`src/wrappers.py`)
This is the most complex part of the Python logic. It creates a bridge between RDDL (Dictionary based) and PyTorch (Vector based).

1.  **`RDDLDecisionWrapper` (The Skipper):**
    * **Problem:** The raw simulator outputs "empty" steps while trains are driving.
    * **Solution:** Uses a `_skip_forward` loop. It steps the environment with `Action=0` repeatedly until a train reaches a **Decision State** (Arriving at station OR Leaving a queue).
    * **Active Stations:** It filters the action dictionary so the agent only controls active stations. All others get `0`.

2.  **`PPOAdapter` (The Translator):**
    * **Input:** Dictionary State (`{'train_timer__t1': 15.0, ...}`).
    * **Output:** Flattened Float Vector.
    * **Crash Fix:** Contains a critical fix for string handling. It checks `if isinstance(val, str)` **before** checking `np.isscalar`, otherwise Numpy crashes on values like `'s1'`.

### B. The Agent (`src/agent.py`)
* **Algorithm:** PPO (Actor-Critic).
* **Network:** 3-layer MLP.
* **Input:** Flattened State Vector.
* **Output:** Discrete Action (0-10).
* **Checkpointing:**
    * Saves dictionary: `{'model_state', 'optimizer_state', 'episode'}`.
    * This allows the training loop to resume the *episode counter* exactly where it left off.

## 4. Historical "Gotchas" & Fixed Bugs
*If modifying code, ensure these regressions are not reintroduced:*

1.  **Variable Renaming:**
    * RDDL `wait_action` was renamed to `wait`.
    * RDDL `wait` (intermediate) was renamed to `delay`.
    * *Risk:* The wrappers manually construct action keys (`f'wait___{s}'`). If the RDDL file changes, the wrapper keys **must** be updated manually.

2.  **Time Tracking:**
    * `global_timer` is a *delta* (time to next event).
    * We explicitly added `current_time` to the RDDL state to track wall-clock time for accurate reward calculation.

3.  **Quadratic Reward:**
    * Old logic summed over `?s: station` checking equality.
    * New logic uses `current_station(?t)` as an index key for $O(1)$ lookup per train.

4.  **String Parsing:**
    * The environment returns station names (e.g., `s1`) as strings/numpy strings. The Adapter has specific logic to strip the 's' and convert to float.

## 5. Current File Structure