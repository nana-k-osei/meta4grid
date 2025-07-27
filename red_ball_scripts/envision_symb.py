import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from raw_state_wrapper import FilteredSymbolicWrapper  # Save the wrapper code as this .py file
import numpy as np

# Create and wrap the environment
env = gym.make("BabyAI-GoToRedBall-v0", render_mode="human")
env = FullyObsWrapper(env)
env = FilteredSymbolicWrapper(env)

# Reset environment and get observation
obs, _ = env.reset()

# Print symbolic observation
print("\n=== Symbolic Observation ===")
for i, obj in enumerate(obs):
    if not np.allclose(obj, 0.0):  # ignore padded zeros
        type_id, color_id, x_pos, y_pos = obj
        print(f"Object {i+1}: type={type_id:.2f}, color={color_id:.2f}, x={x_pos:.2f}, y={y_pos:.2f}")
