import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper

env = gym.make("BabyAI-GoToRedBallGrey-v0", render_mode="ansi")  # or "rgb_array" if you want visuals
env = FullyObsWrapper(env)
obs, _ = env.reset()

print("\n=== Top-Level Observation ===")
print(type(obs))
for key, value in obs.items():
    print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")

print("\n=== Image Grid Sample ===")
print(obs['image'])  # This is usually (7, 7, 3)

print("\n=== Agent Info ===")
print(f"Agent Dir: {obs['direction']}")
print(f"Mission: {obs['mission']}")

print("\n=== From env.unwrapped ===")
base_env = env.unwrapped
print("Agent position:", base_env.agent_pos)
print("Agent direction:", base_env.agent_dir)
print("Grid size:", base_env.width, "x", base_env.height)
print("Goal position:", getattr(base_env, "goal_pos", "N/A"))
print("All objects in grid (if accessible):")
grid = base_env.grid

for x in range(grid.width):
    for y in range(grid.height):
        obj = grid.get(x, y)
        if obj:
            print(f"({x},{y}): {obj}, type={obj.type}, color={obj.color}")
