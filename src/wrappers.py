import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RDDLDecisionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.trains = env.model.type_to_objects['train']
        self.stations = env.model.type_to_objects['station']

        self.TRAIN_IN_ROUTE = 0
        self.TRAIN_IN_QUEUE = 1
        self.TRAIN_WAITING = 2
        self.TRAIN_ACTIVE = 3

        self.full_logs = []
        self.last_obs = None

    def reset(self, **kwargs):
        self.full_logs = []
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()

        self.full_logs.append({
            'step_type': 'RESET',
            'state': obs.copy()
        })

        if not self.get_active_stations(obs):
            obs, _, terminated, truncated, info = self._skip_forward(0, False, False, info)

        return obs, info

    def step(self, action):
        pre_step_obs = self.last_obs.copy()

        # 1. FILTER ACTION
        active_stations = self.get_active_stations(pre_step_obs)

        env_action = {}
        log_action = {}

        for s in self.stations:
            key = f'wait___{s}'

            if s in active_stations:
                val = action.get(key, 0)
                env_action[key] = val
                log_action[key] = val
            else:
                env_action[key] = 0
                log_action[key] = None

        # 2. Apply ENV Action
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.last_obs = obs.copy()

        # Inject LOG Action
        info['filtered_action'] = log_action

        self.full_logs.append({
            'step_type': 'AGENT_ACTION',
            'state': pre_step_obs,
            'action': log_action,
            'original_action': action,
            'next_state': obs.copy(),
            'reward': reward
        })

        # 3. Skipping Loop
        if not terminated and not truncated:
            obs, extra_reward, terminated, truncated, info = self._skip_forward(reward, terminated, truncated, info)
            reward = extra_reward

        info['filtered_action'] = log_action

        return obs, reward, terminated, truncated, info

    def _skip_forward(self, current_reward, terminated, truncated, info):
        """Helper to run the skipping loop."""
        done = terminated or truncated
        total_reward = current_reward
        obs = self.last_obs

        while not done and not self.get_active_stations(obs):

            pre_internal_obs = self.last_obs.copy()
            default_action = {f'wait___{s}': 0 for s in self.stations}

            obs, reward, terminated, truncated, info = self.env.step(default_action)
            self.last_obs = obs.copy()

            self.full_logs.append({
                'step_type': 'INTERNAL_SKIP',
                'state': pre_internal_obs,
                'action': default_action,
                'next_state': obs.copy(),
                'reward': reward
            })

            done = terminated or truncated
            total_reward += reward

        return obs, total_reward, terminated, truncated, info

    def get_active_stations(self, obs):
        active_set = set()
        current_timers = {t: obs[f'train_timer___{t}'] for t in self.trains}
        valid_timers = [t for t in current_timers.values() if t < 1e9]

        if not valid_timers: return set()
        global_timer = min(valid_timers)

        station_occupant = {s: None for s in self.stations}
        station_has_queue = {s: False for s in self.stations}

        for t in self.trains:
            s = obs[f'current_station___{t}']
            state = obs[f'current_state___{t}']
            if state == self.TRAIN_IN_QUEUE:
                station_has_queue[s] = True
            elif state in [self.TRAIN_WAITING, self.TRAIN_ACTIVE]:
                station_occupant[s] = t

        for t in self.trains:
            timer = current_timers[t]
            state = obs[f'current_state___{t}']
            station = obs[f'current_station___{t}']

            if timer == global_timer:
                if state == self.TRAIN_IN_ROUTE:
                    occupant = station_occupant[station]
                    if occupant is None:
                        active_set.add(station)
                    else:
                        occupant_state = obs[f'current_state___{occupant}']
                        occupant_timer = current_timers[occupant]
                        if (occupant_state == self.TRAIN_ACTIVE and
                            occupant_timer == global_timer and
                            not station_has_queue[station]):
                            active_set.add(station)

                if state == self.TRAIN_ACTIVE and station_has_queue[station]:
                    active_set.add(station)

        return active_set

class PPOAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        sample_obs, _ = self.env.reset()
        self.flat_size = self._dict_to_vec(sample_obs).shape[0]
        print(f"PPO Adapter Initialized. Flattened State Size: {self.flat_size}")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.flat_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(11)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._dict_to_vec(obs), info

    def step(self, action_int):
        action_dict = {}
        for s in self.env.stations:
            action_dict[f'wait___{s}'] = int(action_int)
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        return self._dict_to_vec(obs), reward, terminated, truncated, info

    def _dict_to_vec(self, obs_dict):
        values = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            if isinstance(val, str) or np.issubdtype(type(val), np.str_):
                try:
                    num = float(''.join(filter(str.isdigit, str(val))))
                    values.append(num)
                except ValueError: values.append(0.0)
            elif np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0):
                try: values.append(float(val))
                except: values.append(0.0)
            elif isinstance(val, np.ndarray):
                values.extend(val.flatten().astype(float))
            else: values.append(0.0)
        return np.array(values, dtype=np.float32)