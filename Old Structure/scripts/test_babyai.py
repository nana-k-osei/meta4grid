from gymnasium.envs.registration import register
from minigrid.envs.babyai.goto import GoToRedBlueBall

# Register the environment (must be done in this script's process)
register(
    id='BabyAI-GoToRedBlueBall-v0',
    entry_point='minigrid.envs.babyai.goto:GoToRedBlueBall',
)

import gymnasium as gym

env = gym.make("BabyAI-GoToRedBlueBall-v0", render_mode="human")
obs, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
