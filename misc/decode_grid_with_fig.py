from minigrid.envs.babyai.goto import GoToRedBlueBall
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from PIL import Image
from datetime import datetime

# Register environment if not already
try:
    register(
        id='BabyAI-GoToRedBlueBall-v0',
        entry_point='minigrid.envs.babyai.goto:GoToRedBlueBall',
    )
except gym.error.Error:
    pass  # already registered

# Create and wrap environment for full RGB view
env = gym.make('BabyAI-GoToRedBlueBall-v0')
env = FullyObsWrapper(env)
env = RGBImgObsWrapper(env)  # converts symbolic to RGB image

obs, _ = env.reset()
symbolic = obs["image"]  # RGB image now
h, w, _ = symbolic.shape

print(f"\n--- Grid Shape: {h} x {w} ---\n")

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

# Only print non-empty symbolic cells
symbolic_array = env.unwrapped.grid.encode()
for i in range(symbolic_array.shape[0]):
    for j in range(symbolic_array.shape[1]):
        obj_id, color_id, state_id = symbolic_array[i, j]
        obj = IDX_TO_OBJECT.get(obj_id, "?")
        color = IDX_TO_COLOR.get(color_id, "?")
        state = IDX_TO_STATE.get(state_id, "-")
        if obj != "empty":
            print(f"Cell ({i},{j}) → {color} {obj} ({state})")

# Agent metadata
agent_pos = env.unwrapped.agent_pos
agent_dir = env.unwrapped.agent_dir
direction_map = ['→', '↓', '←', '↑']
agent_symbol = direction_map[agent_dir]

print(f"\nDirection (0=right, 1=down, 2=left, 3=up): {obs['direction']}")
print(f"Mission: {obs['mission']}")
print(f"Agent position: {agent_pos}")
print(f"Agent direction: {agent_dir}")

# -------- Save Snapshot --------
img = Image.fromarray(obs["image"])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
img.save(f"env_snapshot_{timestamp}.png")
print(f"\nSnapshot saved as: env_snapshot_{timestamp}.png")
