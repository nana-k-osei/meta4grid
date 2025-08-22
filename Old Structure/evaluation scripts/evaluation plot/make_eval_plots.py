# pip install pandas matplotlib numpy openpyxl
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_DIR = "eval_out"         # folder with your CSV/XLSX files
OUT_DIR = "figs"            # output for plots
os.makedirs(OUT_DIR, exist_ok=True)

# ------- helpers -------
PAT = re.compile(r"(GoToRedBall(Grey)?)_(baseline|macro)_seed(\d+)", re.IGNORECASE)

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    # infer env/agent/seed from filename if missing
    m = PAT.search(os.path.basename(path))
    if m is not None:
        env_name = m.group(1)
        agent = m.group(3).lower()
        seed = int(m.group(4))
        if "env" not in df.columns:   df["env"] = env_name
        if "agent" not in df.columns: df["agent"] = agent
        if "seed" not in df.columns:  df["seed"] = seed
    # sanity columns
    needed = {"episode","reward","length","success","env","agent","seed"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df

def summarise(df):
    # per seed -> summarise, then aggregate across seeds
    per_seed = df.groupby(["env","agent","seed"]).agg(
        success_rate=("success","mean"),
        mean_reward =("reward","mean"),
        avg_steps   =("length","mean"),
    ).reset_index()
    # across seeds
    summary = per_seed.groupby(["env","agent"]).agg(
        success_mean=("success_rate","mean"),
        success_std =("success_rate","std"),
        reward_mean =("mean_reward","mean"),
        reward_std  =("mean_reward","std"),
        steps_mean  =("avg_steps","mean"),
        steps_std   =("avg_steps","std"),
        n=("seed","count")
    ).reset_index()
    return per_seed, summary

def plot_triplet(sub, title, outfile):
    labels = ["Success Rate", "Mean Reward", "Avg Steps"]
    def get(row, m): return row[m+"_mean"], row[m+"_std"]
    base = sub[sub.agent=="baseline"].iloc[0]
    macr = sub[sub.agent=="macro"].iloc[0]
    means_b = [get(base,"success")[0], get(base,"reward")[0], get(base,"steps")[0]]
    stds_b  = [get(base,"success")[1], get(base,"reward")[1], get(base,"steps")[1]]
    means_m = [get(macr,"success")[0], get(macr,"reward")[0], get(macr,"steps")[0]]
    stds_m  = [get(macr,"success")[1], get(macr,"reward")[1], get(macr,"steps")[1]]

    x = np.arange(len(labels)); w = 0.35
    plt.figure(figsize=(7.2, 5.0))
    ax = plt.gca()
    ax.bar(x - w/2, means_b, w, yerr=stds_b, capsize=3, label="Baseline")
    ax.bar(x + w/2, means_m, w, yerr=stds_m, capsize=3, label="Macro")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Value"); ax.set_title(title, pad=10)
    ax.grid(True, linewidth=0.6, alpha=0.3); ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, outfile + ".png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, outfile + ".pdf"))
    plt.close()

def plot_success_combined(summary, outfile="summary_success"):
    env_order = ["GoToRedBall","GoToRedBallGrey"]
    data, xt = [], []
    for env in env_order:
        for agent in ["baseline","macro"]:
            row = summary[(summary.env==env)&(summary.agent==agent)]
            if row.empty: continue
            row = row.iloc[0]
            data.append((row.success_mean, row.success_std))
            tag = "RB" if env=="GoToRedBall" else "RBG"
            xt.append(f"{tag}-{agent.capitalize()}")
    x = np.arange(len(data))
    means = [d[0] for d in data]; stds = [d[1] for d in data]
    plt.figure(figsize=(7.2, 5.0))
    ax = plt.gca()
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(xt)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success Rate"); ax.set_title("Final Success Across Tasks", pad=10)
    ax.grid(True, linewidth=0.6, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, outfile + ".png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, outfile + ".pdf"))
    plt.close()

# ------- run -------
files = sorted(glob.glob(os.path.join(IN_DIR, "*.*")))
if not files:
    raise SystemExit(f"No files found in {IN_DIR}")

dfs = [load_file(p) for p in files]
df_all = pd.concat(dfs, ignore_index=True)
per_seed, summary = summarise(df_all)

# Save summaries
os.makedirs("eval_out_summaries", exist_ok=True)
per_seed.to_csv("eval_out_summaries/per_seed_metrics.csv", index=False)
summary.to_csv("eval_out_summaries/summary_by_env_agent.csv", index=False)
print("Wrote: eval_out_summaries/per_seed_metrics.csv and summary_by_env_agent.csv")

# Per-environment bar charts
for env in sorted(summary.env.unique()):
    sub = summary[summary.env==env]
    if set(sub.agent) >= {"baseline","macro"}:
        short = "rb" if env=="GoToRedBall" else "rbg"
        plot_triplet(sub, f"{env}: Final Evaluation", f"{short}_final_eval")

# Combined success chart
plot_success_combined(summary)
print(f"Wrote figures to {OUT_DIR}")
