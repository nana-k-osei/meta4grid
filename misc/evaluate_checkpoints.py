import gymnasium as gym
import minigrid
import numpy as np
from stable_baselines3 import PPO
from minigrid.wrappers import FullyObsWrapper
from raw_state_wrapper import RawStateWrapper, SymbolicObsWrapper, FilteredSymbolicWrapper, MacroRawStateWrapper
from macro_wrapper import RewardTracker, UnstuckHandler  # Macro wrappers 


def run_multiple_evals(model, env_fn, num_runs=5, episodes_per_run=100):
    results = []
    for i in range(num_runs):
        env = env_fn()
        success_count = 0
        step_total = 0
        truncated_count = 0
        failed_terminated_count = 0
        episode_rewards = []

        for _ in range(episodes_per_run):
            obs, _ = env.reset()
            done = False
            steps = 0
            reward_total = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                reward_total += reward
                steps += 1

            step_total += steps
            episode_rewards.append(reward_total)

            if reward_total > 0:
                success_count += 1
            elif truncated:
                truncated_count += 1
            else:
                failed_terminated_count += 1

        success_rate = success_count / episodes_per_run
        avg_steps = step_total / episodes_per_run
        avg_reward = np.mean(episode_rewards)
        results.append((success_rate, avg_steps, avg_reward))

        print(f"\nRun {i+1}:")
        print(f"  Successes     : {success_count}")
        print(f"  Timeouts      : {truncated_count}")
        print(f"  Failures      : {failed_terminated_count}")
        print(f"  Success Rate  : {success_rate:.2f}")
        print(f"  Avg Steps     : {avg_steps:.1f}")
        print(f"  Avg Reward    : {avg_reward:.3f}")
        env.close()

    # Summary
    sr_values = [r[0] for r in results]
    steps_values = [r[1] for r in results]
    reward_values = [r[2] for r in results]
    print(f"\n=== Summary over {num_runs} runs ===")
    print(f"Mean Success Rate: {np.mean(sr_values):.3f} ± {np.std(sr_values):.3f}")
    print(f"Mean Avg Steps  : {np.mean(steps_values):.1f} ± {np.std(steps_values):.1f}")
    print(f"Mean Avg Reward : {np.mean(reward_values):.3f} ± {np.std(reward_values):.3f}")


def make_env():
    env = gym.make("BabyAI-GoToRedBall-v0")
    env = FullyObsWrapper(env)
    #env = RawStateWrapper(env)
    env = RewardTracker(env)
    env = UnstuckHandler(env)
    env = MacroRawStateWrapper(env)
    return env


if __name__ == "__main__":
    model = PPO.load("./red_ball_playground/models/ppo_redball_macro_final")  # trained RedBall model
    run_multiple_evals(model, make_env, num_runs=5, episodes_per_run=100)
