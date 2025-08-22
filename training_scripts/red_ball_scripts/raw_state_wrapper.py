import gymnasium as gym
import numpy as np

# Minimal wrapper to convert MiniGrid observations to raw symbolic state
class RawStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.width  
        # Features: agent_x, agent_y, agent_dir, red_x, red_y
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

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

        obs_vec = np.array([agent_x, agent_y, agent_dir_norm, *red_pos], dtype=np.float32)
        return obs_vec

# Raw state wrapper for macro-actions
class MacroRawStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.width
        # Features: agent_x, agent_y, agent_dir, red_x, red_y, red_type, red_color
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

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

        obs_vec = np.array([
            agent_x, agent_y,
            agent_dir_norm,
            *red_pos,
            red_type,
            red_color
        ], dtype=np.float32)

        return obs_vec



# Full symbolic observation wrapper
class SymbolicObsWrapper(gym.ObservationWrapper):
    """
    Converts the full 8x8x3 MiniGrid image into a symbolic list of objects with type, color, and (x, y) location.
    Each object is encoded as (type_id, color_id, x/grid_size, y/grid_size)
    """
    def __init__(self, env):
        super().__init__(env)
        self.grid_size = self.env.unwrapped.width  # typically 8
        self.max_objects = self.grid_size ** 2      # worst-case: all cells filled
        # Each object: type_id, color_id, x, y
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.max_objects, 4),
            dtype=np.float32
        )

    def observation(self, obs):
        image = obs["image"]  # shape: (grid_size, grid_size, 3)
        objects = []

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                obj_type, obj_color, _ = image[x, y]
                if obj_type != 0:  # Skip empty cells
                    obj_vec = [
                        obj_type / 10.0,    # normalize type_id
                        obj_color / 10.0,   # normalize color_id
                        x / self.grid_size,
                        y / self.grid_size
                    ]
                    objects.append(obj_vec)

        # Pad with zeros to always return a fixed-size array
        while len(objects) < self.max_objects:
            objects.append([0.0, 0.0, 0.0, 0.0])

        return np.array(objects, dtype=np.float32)

# Filtered symbolic observation wrapper
class FilteredSymbolicWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.grid_size = self.env.unwrapped.width  # Typically 8x8
        self.max_objects = self.grid_size ** 2  # max possible
        self.object_dim = 4  # type, color, x, y

        # Observation shape = (max_objects, 4)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.max_objects, self.object_dim), dtype=np.float32
        )

    def observation(self, obs):
        grid = obs["image"]  # shape (grid_size, grid_size, 3)
        symbolic_obs = []

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                obj_type, color, state = grid[x, y]

                # Filter relevant objects: ball(6), key(5), box(7)
                if obj_type in [5, 6, 7]:
                    symbolic_obs.append([
                        obj_type / 10.0,         # normalize type
                        color / 10.0,            # normalize color
                        x / self.grid_size,      # normalize position
                        y / self.grid_size
                    ])

        # Pad with zeros if fewer than max_objects
        while len(symbolic_obs) < self.max_objects:
            symbolic_obs.append([0.0, 0.0, 0.0, 0.0])

        return np.array(symbolic_obs, dtype=np.float32)
