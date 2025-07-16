import gymnasium as gym
import minigrid
import numpy as np
from stable_baselines3 import PPO
from raw_state_wrapper import RawStateWrapper


# Load trained model
model = PPO.load("ppo_baseline_redblueball")

# Set up eval environment
def make_env():
    env = gym.make("BabyAI-GoToRedBlueBall-v0", render_mode=None)
    env = RawStateWrapper(env)
    return env

env = make_env()

# Evaluation loop
n_episodes = 100
successes = 0
total_steps = []

for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    total_steps.append(steps)
    if reward > 0:
        successes += 1

# Print results
success_rate = successes / n_episodes
avg_steps = np.mean(total_steps)

print(f"\nEvaluated over {n_episodes} episodes:")
print(f" Success rate: {success_rate:.2f}")
print(f" Average steps per episode: {avg_steps:.1f}")

