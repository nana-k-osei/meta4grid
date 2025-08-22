from minigrid.envs.babyai.goto import GoToRedBlueBall
from minigrid.wrappers import FullyObsWrapper
from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

# Register environment if not already
try:
    register(
        id='BabyAI-GoToRedBlueBall-v0',
        entry_point='minigrid.envs.babyai.goto:GoToRedBlueBall',
    )
except gym.error.Error:
    pass  # already registered

# Create and wrap environment
env = gym.make('BabyAI-GoToRedBlueBall-v0')
env = FullyObsWrapper(env)

obs, _ = env.reset()
symbolic = obs["image"]
h, w, _ = symbolic.shape

print(f"\n--- Grid Shape: {h} x {w} ---\n")

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

# Only print non-empty cells
for i in range(h):
    for j in range(w):
        obj_id, color_id, state_id = symbolic[i, j]
        obj = IDX_TO_OBJECT.get(obj_id, "?")
        color = IDX_TO_COLOR.get(color_id, "?")
        state = IDX_TO_STATE.get(state_id, "-")

        if obj != "empty":
            print(f"Cell ({i},{j}) → {color} {obj} ({state})")

# Show direction and mission
#print(f"\nDirection (0=right, 1=down, 2=left, 3=up): {obs['direction']}")
#print(f"Mission: {obs['mission']}")
print(f"\nDirection (0=right, 1=down, 2=left, 3=up): {obs['direction']}")
print(f"Mission: {obs['mission']}")
print(f"Agent position: {env.unwrapped.agent_pos}")
print(f"Agent direction: {env.unwrapped.agent_dir}")


