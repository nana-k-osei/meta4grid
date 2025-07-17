import gymnasium as gym
import numpy as np

# Minimal wrapper to convert MiniGrid observations to raw symbolic state
class RawStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # agent_x, agent_y, agent_dir, red_x, red_y, blue_x, blue_y
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

    def observation(self, obs):
        grid = obs["image"]           # (7, 7, 3)
        direction = obs["direction"]  # 0–3
        #agent_pos = np.array(self.env.agent_pos, dtype=np.float32)
        base_env = self.env.unwrapped
        agent_pos = np.array(base_env.agent_pos, dtype=np.float32)
        #grid_size = self.env.width    # Usually 7
        grid_size = base_env.width

        # Normalize agent position and direction
        agent_x, agent_y = agent_pos / grid_size
        agent_dir = direction / 3.0

        # Default positions if objects not visible
        red_pos = np.zeros(2)
        blue_pos = np.zeros(2)

        # Scan grid for red and blue balls
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                obj_id, color_id, state = grid[x, y]
                if obj_id == 6:  # 6 = ball
                    if color_id == 0:  # red
                        red_pos = np.array([x, y]) / grid_size
                    elif color_id == 1:  # blue
                        blue_pos = np.array([x, y]) / grid_size

        obs_vec = np.array([
            agent_x, agent_y,
            agent_dir,
            *red_pos,
            *blue_pos
        ], dtype=np.float32)

        return obs_vec

# Richer version of the RawStateWrapper with more symbolic features
class RawStateWrapperRich(gym.ObservationWrapper):
    """
    Adds richer symbolic features:
    - Agent position (x, y)
    - Agent direction (normalized)
    - Red ball visibility (binary)
    - Blue ball visibility (binary)
    - Normalized distance to red and blue (0 if not visible)
    """
    def __init__(self, env):
        super().__init__(env)
        # 2 agent pos + 1 direction + 2 visibility flags + 2 distances = 7 features
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

    def observation(self, obs):
        base_env = self.env.unwrapped
        grid = obs["image"]            # shape (7, 7, 3)
        direction = obs["direction"]
        agent_pos = np.array(base_env.agent_pos, dtype=np.float32)
        grid_size = base_env.width     # Usually 7

        # Normalize agent position and direction
        agent_x, agent_y = agent_pos / grid_size
        agent_dir = direction / 3.0

        # Init values
        red_visible, blue_visible = 0.0, 0.0
        dist_to_red, dist_to_blue = 0.0, 0.0

        # Agent's view center is at (3, 3) in obs["image"]
        view_center = np.array([3, 3])

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                obj_id, color_id, _ = grid[x, y]
                if obj_id == 6:  # ball
                    rel_pos = np.array([x, y])
                    dist = np.linalg.norm(rel_pos - view_center) / (np.sqrt(2) * grid.shape[0])  # normalize

                    if color_id == 0:  # red
                        red_visible = 1.0
                        dist_to_red = dist
                    elif color_id == 1:  # blue
                        blue_visible = 1.0
                        dist_to_blue = dist

        obs_vec = np.array([
            agent_x, agent_y,
            agent_dir,
            red_visible, blue_visible,
            dist_to_red, dist_to_blue
        ], dtype=np.float32)

        return obs_vec