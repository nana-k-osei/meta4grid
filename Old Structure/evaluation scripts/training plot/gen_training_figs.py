from tbparse import SummaryReader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# ---------- helpers ----------
def load_tb_scalars(log_dir, tag):
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.scalars[["step", tag]].dropna()
    df = df.sort_values("step")
    return df

def rolling_mean(y, w=201):
    if w <= 1 or w > len(y):
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

def fmt_xticks_1e7(x, _):
    return f"{x/1e7:.1f}"

def plot_pair(baseline_df, macro_df, *, title, ylabel, out_png,
              tag="rollout/ep_rew_mean", smooth=201, ylim=(0.1, 0.9)):
    os.makedirs("figs", exist_ok=True)

    xb, yb = baseline_df["step"].to_numpy(), baseline_df[tag].to_numpy()
    xm, ym = macro_df["step"].to_numpy(),    macro_df[tag].to_numpy()

    yb = rolling_mean(yb, smooth)
    ym = rolling_mean(ym, smooth)

    plt.figure(figsize=(7.2, 5.0))  # good single-column size
    ax = plt.gca()
    ax.plot(xb, yb, label="Baseline", linewidth=2.0)
    ax.plot(xm, ym, label="Macro",    linewidth=2.0)

    ax.set_title(title, pad=10)
    ax.set_xlabel(r"Timesteps ($\times 10^{7}$)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_xticks_1e7))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linewidth=0.6, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_png.replace(".png", ".pdf"))
    plt.close()

# ---------- config ----------
TAG = "rollout/ep_rew_mean"   # use 'eval/mean_reward' 
SMOOTH = 201                  # reduce to 51–101 for sparse logs
YLIM = (0.1, 0.9)             # keep identical across tasks for comparability

# ---------- RedBall ----------
rb_base  = load_tb_scalars("../../eval_logs/redball/redball_baseline", TAG)
rb_macro = load_tb_scalars("../../eval_logs/redball/redball_macro",    TAG)
plot_pair(rb_base, rb_macro,
          title="RedBall: Mean Episodic Reward",
          ylabel="Mean Episodic Reward",
          out_png="figs/rb_reward_curve.png",
          tag=TAG, smooth=SMOOTH, ylim=YLIM)

# ---------- RedBallGrey ----------
rbg_base  = load_tb_scalars("../../eval_logs/redballgrey/redballgrey_baseline", TAG)
rbg_macro = load_tb_scalars("../../eval_logs/redballgrey/redballgrey_macro",    TAG)
plot_pair(rbg_base, rbg_macro,
          title="RedBallGrey: Mean Episodic Reward",
          ylabel="Mean Episodic Reward",
          out_png="figs/rbg_reward_curve.png",
          tag=TAG, smooth=SMOOTH, ylim=YLIM)
