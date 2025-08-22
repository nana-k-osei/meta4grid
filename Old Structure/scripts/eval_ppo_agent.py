import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

# Load environment
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
env = ImgObsWrapper(env)

# Load model
#model = PPO.load("ppo_minigrid_empty8x8")
model = PPO.load("ppo_minigrid_empty8x8_v2", env=env)  # Ensure the environment is passed for correct observation space

# Run a few episodes
for episode in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Episode {episode + 1} finished with reward: {total_reward}")
env.close()
