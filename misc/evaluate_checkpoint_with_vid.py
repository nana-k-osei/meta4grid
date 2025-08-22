import gymnasium as gym
import numpy as np
import minigrid
import time
from stable_baselines3 import PPO
from minigrid.wrappers import FullyObsWrapper
from raw_state_wrapper import RawStateWrapper, SymbolicObsWrapper, FilteredSymbolicWrapper, MacroRawStateWrapper
from macro_wrapper import RewardTracker, UnstuckHandler  # Macro wrappers 
from gymnasium.wrappers import RecordVideo
from datetime import datetime


def make_env_with_video(run_idx):
    def _init():
        env = gym.make("BabyAI-GoToRedBallGrey-v0", render_mode="rgb_array")
        env = FullyObsWrapper(env)
        env = RawStateWrapper(env)
        #env = RewardTracker(env)
        #env = UnstuckHandler(env)
        #env = MacroRawStateWrapper(env)
        video_folder = f"videos/redballgrey_vids/redballgrey_run_{run_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
        return env
    return _init


def evaluate_and_record(model, num_episodes):
    for i in range(num_episodes):
        env = make_env_with_video(i)()
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()  
            done = terminated or truncated
            total_reward += reward
            steps += 1

        print(f"Episode {i+1} finished: Steps = {steps}, Reward = {total_reward:.4f}")
        env.close()

    print("\n Videos saved under `videos/` folder.")


if __name__ == "__main__":
    model = PPO.load("../../red_ball_grey_playground/models/ppo_redballgrey_baseline")

    start = time.time()
    evaluate_and_record(model, num_episodes=5)
    end = time.time()

    minutes, seconds = divmod(int(end - start), 60)
    print(f"\nEvaluation and video recording completed in {minutes}m {seconds}s")

