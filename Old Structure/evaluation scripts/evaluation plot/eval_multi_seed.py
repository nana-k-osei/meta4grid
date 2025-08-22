import os, csv, argparse, numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import FullyObsWrapper
from raw_state_wrapper import RawStateWrapper, SymbolicObsWrapper, FilteredSymbolicWrapper, MacroRawStateWrapper
from macro_wrapper import RewardTracker, UnstuckHandler  # Macro wrappers

def make_env(env_id: str, agent: str):
    env = gym.make(env_id)
    env = FullyObsWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=64)
    env = Monitor(env)
    if agent == "macro":
        env = RewardTracker(env)
        env = UnstuckHandler(env)
        env = MacroRawStateWrapper(env)
    else:
        env = RawStateWrapper(env)
    return env

def eval_once(env, model, episodes: int):
    rows = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False; truncated = False
        ep_rew = 0.0; ep_len = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = env.step(action)
            ep_rew += float(r); ep_len += 1
        success = 1 if ep_rew > 0 else 0
        rows.append({"episode": ep, "reward": ep_rew, "length": ep_len, "success": success})
    return rows

def write_csv(rows, out_dir, env_name, agent, seed):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{env_name}_{agent}_seed{seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode","reward","length","success"])
        w.writeheader(); w.writerows(rows)
    return path

def summarise(rows):
    r = np.array([x["reward"] for x in rows], float)
    s = np.array([x["success"] for x in rows], float)
    l = np.array([x["length"] for x in rows], float)
    return dict(success_rate=float(s.mean()), mean_reward=float(r.mean()), avg_steps=float(l.mean()))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True, help="BabyAI-GoToRedBall-v0 or BabyAI-GoToRedBallGrey-v0")
    p.add_argument("--model_path", required=True, help="Path to PPO .zip")
    p.add_argument("--agent", choices=["baseline","macro"], required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])  # 5 seeds recommended
    p.add_argument("--out_dir", default="eval_out")
    args = p.parse_args()

    all_summ = []
    for seed in args.seeds:
        env = make_env(args.env_id, args.agent)
        env.reset(seed=seed)
        model = PPO.load(args.model_path, env=env, device="cpu")  # or "cuda" if available
        rows = eval_once(env, model, args.episodes)
        csv_path = write_csv(rows, args.out_dir, args.env_id.replace("BabyAI-","").replace("-v0",""), args.agent, seed)
        summ = summarise(rows); summ["seed"] = seed
        all_summ.append(summ)
        print(f"[{args.env_id} | {args.agent} | seed {seed}] "
              f"success={summ['success_rate']:.3f}, reward={summ['mean_reward']:.3f}, steps={summ['avg_steps']:.1f} -> {csv_path}")

    # Print mean ± std over seeds
    S = {k: np.array([d[k] for d in all_summ], float) for k in ["success_rate","mean_reward","avg_steps"]}
    print("\n== Summary over seeds ==")
    for k in ["success_rate","mean_reward","avg_steps"]:
        print(f"{k}: {S[k].mean():.3f} ± {S[k].std():.3f}  (n={len(S[k])})")
