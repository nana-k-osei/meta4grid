import gymnasium as gym
import numpy as np
import time
import minigrid  # ensure BabyAI envs are registered
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from raw_state_wrapper import RawStateWrapper
from raw_state_wrapper import SymbolicObsWrapper
from raw_state_wrapper import FilteredSymbolicWrapper
from minigrid.wrappers import FullyObsWrapper
from gymnasium.wrappers import TimeLimit


def make_env():
    def _init():
        env = gym.make("BabyAI-GoToRedBall-v0")
        env = FullyObsWrapper(env)
        env = TimeLimit(env, max_episode_steps=64)  # Cap episode steps
        env = Monitor(env)
        env = RawStateWrapper(env) 
        #env = SymbolicObsWrapper(env) 
        #env = FilteredSymbolicWrapper(env) 
        return env
    return _init


if __name__ == "__main__":
    n_envs = 8 # Number of parallel environments
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./red_ball_playground/ppo_redball_logs/",
        device="cpu"
    )

    start = time.time()
    total_timesteps = 15_000_000
    model.learn(total_timesteps=total_timesteps)
    end = time.time()

    minutes, seconds = divmod(int(end - start), 60)
    print(f"\n PPO on GoToRedBall for {total_timesteps} total timesteps completed in {minutes}m {seconds}s")

    model.save("./red_ball_playground/models/ppo_redball_filtered_baseline")  # Save final model
