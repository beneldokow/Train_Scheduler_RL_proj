import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RDDLDecisionWrapper(gym.Wrapper):
    """
    Decision-Point Skipping Wrapper for RDDL Environments.
    
    RDDL simulations often have many "no-op" steps where no agent action is required 
    (e.g., while a train is driving between stations). This wrapper automatically 
    fast-forwards through these steps until a train reaches a station and a decision 
    needs to be made.
    
    This significantly reduces the complexity of the RL problem by focusing only on 
    meaningful state-action pairs.
    """
    def __init__(self, env):
        super().__init__(env)
        # Type-to-object mapping for state extraction
        self.trains = env.model.type_to_objects["train"]
        self.stations = env.model.type_to_objects["station"]

        # Local copies of domain constants
        self.TRAIN_IN_ROUTE = 0
        self.TRAIN_IN_QUEUE = 1
        self.TRAIN_WAITING = 2
        self.TRAIN_ACTIVE = 3

        self.full_logs = []
        self.last_obs = None

    def reset(self, **kwargs):
        """Resets the environment and skips to the first decision point."""
        self.full_logs = []
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()

        # Check if we need to skip forward immediately after reset
        if not self.get_active_stations(obs):
            obs, _, terminated, truncated, info = self._skip_forward(0, False, False, info)

        return obs, info

    def step(self, action):
        """Applies the agent's action and fast-forwards to the next decision point."""
        pre_step_obs = self.last_obs.copy()

        # 1. Action Filtering: Map the high-level action only to stations requiring a decision
        active_stations = self.get_active_stations(pre_step_obs)
        env_action = {}
        for s in self.stations:
            key = f"wait___{s}"
            env_action[key] = action.get(key, 0) if s in active_stations else 0

        # 2. Step the base environment
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.last_obs = obs.copy()

        # 3. Skipping Loop: Automate no-op steps until the next decision point
        if not terminated and not truncated:
            obs, extra_reward, terminated, truncated, info = self._skip_forward(reward, terminated, truncated, info)
            reward = extra_reward # Accumulate rewards during the skip

        return obs, reward, terminated, truncated, info

    def _skip_forward(self, current_reward, terminated, truncated, info):
        """Internal helper to step the environment with default actions until a decision is needed."""
        done = terminated or truncated
        total_reward = current_reward
        obs = self.last_obs

        # Keep stepping until a terminal state or an active station is found
        while not done and not self.get_active_stations(obs):
            default_action = {f"wait___{s}": 0 for s in self.stations}
            obs, reward, terminated, truncated, info = self.env.step(default_action)
            self.last_obs = obs.copy()
            done = terminated or truncated
            total_reward += reward

        return obs, total_reward, terminated, truncated, info

    def get_active_stations(self, obs):
        """
        Determines which stations currently require a 'wait' decision.
        
        A station is active if:
        1. A train is arriving from 'IN_ROUTE' and the platform is clear.
        2. A train is currently 'ACTIVE' (boarding) but there is a queue waiting.
        """
        active_set = set()
        current_timers = {t: obs[f"train_timer___{t}"] for t in self.trains}
        
        # Determine the global event horizon (the next time an event will occur)
        valid_timers = [t for t in current_timers.values() if t < 1e9]
        if not valid_timers: return set()
        global_timer = min(valid_timers)

        # Track station occupancy
        station_occupant = {s: None for s in self.stations}
        station_has_queue = {s: False for s in self.stations}
        for t in self.trains:
            s = obs[f"current_station___{t}"]
            state = obs[f"current_state___{t}"]
            if state == self.TRAIN_IN_QUEUE: station_has_queue[s] = True
            elif state in [self.TRAIN_WAITING, self.TRAIN_ACTIVE]: station_occupant[s] = t

        # Core logic for identifying decision points
        for t in self.trains:
            timer, state, station = current_timers[t], obs[f"current_state___{t}"], obs[f"current_station___{t}"]

            if timer == global_timer:
                # Case 1: Train reaching a station
                if state == self.TRAIN_IN_ROUTE:
                    if station_occupant[station] is None: active_set.add(station)
                
                # Case 2: Train finishing boarding while others are waiting in queue
                if state == self.TRAIN_ACTIVE and station_has_queue[station]:
                    active_set.add(station)

        return active_set


class PPOAdapter(gym.Wrapper):
    """
    PPO Adapter for RDDL Environments.
    
    PyTorch PPO implementations expect:
    1. Observations as flat numerical vectors (np.ndarray).
    2. Discrete action spaces (int) for categorical policies.
    
    This wrapper handles the conversion between RDDL dictionaries and these standard formats.
    """
    def __init__(self, env):
        super().__init__(env)
        # Determine observation size by performing a trial vectorization
        sample_obs, _ = self.env.reset()
        self.flat_size = self._dict_to_vec(sample_obs).shape[0]
        
        # Standard Gym Space Definitions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.flat_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(11) # Maps to 0-10 wait minutes

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._dict_to_vec(obs), info

    def step(self, action_int):
        """Maps a single discrete action to wait actions for all stations."""
        action_dict = {f"wait___{s}": int(action_int) for s in self.env.stations}
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        return self._dict_to_vec(obs), reward, terminated, truncated, info

    def _dict_to_vec(self, obs_dict):
        """
        Converts a multi-modal RDDL observation dictionary to a flat numerical vector.
        
        - Extracts numbers from object-strings (e.g., 's1' -> 1.0).
        - Flattens arrays.
        - Handles scalars.
        """
        values = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            # Case 1: Object identifiers (e.g., station names)
            if isinstance(val, str) or np.issubdtype(type(val), np.str_):
                try:
                    num = float("".join(filter(str.isdigit, str(val))))
                    values.append(num)
                except ValueError: values.append(0.0)
            # Case 2: Scalars
            elif np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0):
                values.append(float(val))
            # Case 3: Multidimensional arrays
            elif isinstance(val, np.ndarray):
                values.extend(val.flatten().astype(float))
            else: values.append(0.0)
        return np.array(values, dtype=np.float32)
