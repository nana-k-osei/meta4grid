import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from raw_state_wrapper import RawStateWrapper  # Your custom wrapper

import gymnasium as gym

def make_env():
    env = gym.make("BabyAI-GoToRedBlueBall-v0")
    env = Monitor(env)
    env = RawStateWrapper(env)
    return env

# Wrap in DummyVecEnv for SB3 compatibility
env = DummyVecEnv([make_env])

# Train PPO agent on symbolic state input
model = PPO(
    "MlpPolicy",              # Use MLP, not CNN
    env,
    verbose=1,
    tensorboard_log="./ppo_baseline_logs/",
    device="cpu"  # Use CPU for training
)

model.learn(total_timesteps=300_000)

# Save the trained model
model.save("ppo_baseline_redblueball")
