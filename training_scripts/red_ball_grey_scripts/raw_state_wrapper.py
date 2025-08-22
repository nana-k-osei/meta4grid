import gymnasium as gym
import numpy as np


# Minimal wrapper to convert MiniGrid observations to raw symbolic state
class RawStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.width
        # Features: agent_x, agent_y, agent_dir, red_x, red_y
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )

    def observation(self, obs):
        base_env = self.env.unwrapped
        grid = obs["image"]  # (grid_size, grid_size, 3)
        agent_pos = np.array(base_env.agent_pos, dtype=np.float32)
        agent_dir = obs["direction"]  # int in [0, 3]

        # Normalize agent position and direction
        agent_x, agent_y = agent_pos / self.grid_size
        agent_dir_norm = agent_dir / 3.0

        # Default: no red ball found
        red_pos = np.zeros(2)

        # Search grid for red ball (obj_id=6, color_id=0)
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                obj_id, color_id, _ = grid[x, y]
                if obj_id == 6 and color_id == 0:
                    red_pos = np.array([x, y]) / self.grid_size
                    break  # Take the first one found

        obs_vec = np.array(
            [agent_x, agent_y, agent_dir_norm, *red_pos], dtype=np.float32
        )
        return obs_vec


# Raw state wrapper for macro-actions
class MacroRawStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.width
        # Features: agent_x, agent_y, agent_dir, red_x, red_y, red_type, red_color
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.float32
        )

    def observation(self, obs):
        base_env = self.env.unwrapped
        grid = obs["image"]  # (grid_size, grid_size, 3)
        agent_pos = np.array(base_env.agent_pos, dtype=np.float32)
        agent_dir = obs["direction"]

        # Normalize agent features
        agent_x, agent_y = agent_pos / self.grid_size
        agent_dir_norm = agent_dir / 3.0

        red_pos = np.zeros(2)
        red_type = 0.0
        red_color = 0.0

        # Scan grid for the first red ball (object ID 6, color ID 0)
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                obj_id, color_id, _ = grid[x, y]
                if obj_id == 6 and color_id == 0:
                    red_pos = np.array([x, y]) / self.grid_size
                    red_type = obj_id / 10.0  # normalize obj_id (10 is max in MiniGrid)
                    red_color = color_id / 5.0  # normalize color_id (5 common colors)
                    break

        obs_vec = np.array(
            [agent_x, agent_y, agent_dir_norm, *red_pos, red_type, red_color],
            dtype=np.float32,
        )

        return obs_vec
