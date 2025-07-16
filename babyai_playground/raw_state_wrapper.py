import gymnasium as gym
import numpy as np

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

