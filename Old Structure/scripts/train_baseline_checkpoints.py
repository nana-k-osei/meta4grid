import gymnasium as gym
import numpy as np
import time
import minigrid  # ensure BabyAI envs are registered
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from raw_state_wrapper import RawStateWrapper  # or your rich version
# from raw_state_wrapper import RawStateWrapperRich
from minigrid.wrappers import FullyObsWrapper
from gymnasium.wrappers import TimeLimit


def make_env():
    def _init():
        env = gym.make("BabyAI-GoToRedBall-v0")
        env = FullyObsWrapper(env)
        env = TimeLimit(env, max_episode_steps=64)  # Cap episode steps
        env = Monitor(env)
        env = RawStateWrapper(env)  # or use RawStateWrapperRich
        return env
    return _init


if __name__ == "__main__":
    n_envs = 8 # Number of parallel environments
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_redball_logs/",
        device="cpu"
    )

    # Save model every 5,000,000 steps
    #checkpoint_callback = CheckpointCallback(
    #    save_freq=5_000_000 // n_envs,
    #    save_path="./checkpoints/",
    #    name_prefix="ppo_redball_v1"
    #)

    start = time.time()
    #model.learn(total_timesteps=15_000_000, callback=checkpoint_callback)
    model.learn(total_timesteps=30_000_000)
    end = time.time()

    minutes, seconds = divmod(int(end - start), 60)
    print(f"\n PPO on GoToRedBall completed in {minutes}m {seconds}s")

    model.save("models/ppo_redball_final_v3")  # Save final model
