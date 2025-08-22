import os
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

# agent color codes
agents = ["Baseline", "Macro"]
colors = ["steelblue", "darkorange"]

# Auto-y limits depending on metric
def get_ylim(metric):
    if metric == "steps":
        return (0, 40)  # tighter range since values < 30
    else:
        return (0, 1)   # success and reward are percentages/proportions

# Function to plot bar charts
def plot_metric(env, metric, values, errors, ylabel, filename):
    fig, ax = plt.subplots(figsize=(5, 4.8))
    bars = ax.bar(agents, values, yerr=errors, capsize=5, color=colors)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Agent")
    ax.set_ylim(*get_ylim(metric))
    ax.set_title(env)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# Generate plots
plot_metric("GoToRedBall: Success Rate per Agent", "success",
            results["GoToRedBall"]["success"][0],
            results["GoToRedBall"]["success"][1],
            "Success rate", f"{OUTDIR}/rb_success.pdf")

plot_metric("GoToRedBall: Mean Reward per Agent", "reward",
            results["GoToRedBall"]["reward"][0],
            results["GoToRedBall"]["reward"][1],
            "Mean reward", f"{OUTDIR}/rb_reward.pdf")

plot_metric("GoToRedBall: Average Steps (64 max steps)", "steps",
            results["GoToRedBall"]["steps"][0],
            results["GoToRedBall"]["steps"][1],
            "Average steps", f"{OUTDIR}/rb_steps.pdf")

plot_metric("GoToRedBallGrey: Success Rate per Agent", "success",
            results["GoToRedBallGrey"]["success"][0],
            results["GoToRedBallGrey"]["success"][1],
            "Success rate", f"{OUTDIR}/rbg_success.pdf")

plot_metric("GoToRedBallGrey: Mean Reward per Agent", "reward",
            results["GoToRedBallGrey"]["reward"][0],
            results["GoToRedBallGrey"]["reward"][1],
            "Mean reward", f"{OUTDIR}/rbg_reward.pdf")

plot_metric("GoToRedBallGrey: Average Steps (64 max steps)", "steps",
            results["GoToRedBallGrey"]["steps"][0],
            results["GoToRedBallGrey"]["steps"][1],
            "Average steps", f"{OUTDIR}/rbg_steps.pdf")

print("All plots saved successfully.")
