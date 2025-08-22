import os
import numpy as np
import matplotlib.pyplot as plt

# summary stats (mean, sd) from your multi-seed evaluation
results = {
    "GoToRedBall": {
        "success": ([0.594, 0.952], [0.039, 0.015]),
        "reward": ([0.561, 0.791], [0.037, 0.013]),
        "steps": ([28.338, 14.526], [2.338, 0.863])
    },
    "GoToRedBallGrey": {
        "success": ([0.572, 0.956], [0.044, 0.026]),
        "reward": ([0.539, 0.767], [0.041, 0.020]),
        "steps": ([29.732, 16.224], [2.615, 1.270])
    }
}

OUTDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)

agents = ["Baseline", "Macro"]
colors = ["steelblue", "darkorange"]

# Combined Success + Reward plot
def plot_success_reward(env, success_vals, success_errs, reward_vals, reward_errs, filename):
    x = np.arange(len(agents))  # positions for agents
    width = 0.35  # bar width
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars1 = ax.bar(x - width/2, success_vals, width, yerr=success_errs, 
                   capsize=5, label="", color=colors[0])
    bars2 = ax.bar(x + width/2, reward_vals, width, yerr=reward_errs, 
                   capsize=5, label="Mean Reward", color=colors[1])

    ax.set_ylabel("Value (0–1)")
    ax.set_xlabel("Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylim(0, 1)
    ax.set_title(env)
    ax.legend()

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

# Steps plot
def plot_steps(env, values, errors, filename):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(agents, values, yerr=errors, capsize=5, color=colors)
    ax.set_ylabel("Average steps")
    ax.set_xlabel("Agent")
    ax.set_ylim(0, 40)  # tighter ceiling
    ax.set_title(env)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate combined plots
plot_success_reward("GoToRedBall",
                    results["GoToRedBall"]["success"][0],
                    results["GoToRedBall"]["success"][1],
                    results["GoToRedBall"]["reward"][0],
                    results["GoToRedBall"]["reward"][1],
                    f"{OUTDIR}/rb_success_reward.pdf")

plot_steps("GoToRedBall",
           results["GoToRedBall"]["steps"][0],
           results["GoToRedBall"]["steps"][1],
           f"{OUTDIR}/rb_steps.pdf")

plot_success_reward("GoToRedBallGrey",
                    results["GoToRedBallGrey"]["success"][0],
                    results["GoToRedBallGrey"]["success"][1],
                    results["GoToRedBallGrey"]["reward"][0],
                    results["GoToRedBallGrey"]["reward"][1],
                    f"{OUTDIR}/rbg_success_reward.pdf")

plot_steps("GoToRedBallGrey",
           results["GoToRedBallGrey"]["steps"][0],
           results["GoToRedBallGrey"]["steps"][1],
           f"{OUTDIR}/rbg_steps.pdf")

print("All plots saved successfully.")
