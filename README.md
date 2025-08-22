# meta4grid
Hierarchical reasoning and high-level action training in MiniGrid environments.

# Macro-augmented PPO for MiniGrid

This repository contains the code for my MSc dissertation on meta-actions in interactive environments. The project studies whether a simple macro component can improve a PPO agent on two BabyAI tasks: **GoToRedBall** and **GoToRedBallGrey**.

**Key idea in one line:** an unstuck macro that breaks loops in procedurally generated maps helps the agent explore more effectively. A reward-tracking module records events for analysis only.

---

## Folder & File Structure 📁

<details>
<summary><strong>evaluation scripts</strong> 📊</summary>

- **initial_evaluators** 📂
  - redball 📄
  - redballgrey 📄
- **plotters** 📂
  - evaluation plot 📄
  - training plot 📄
</details>

<details>
<summary><strong>misc</strong> 🛠️</summary>

- decode_grid.py 📄
- decode_grid_with_fig.py 📄
- envision_symb.py 📄
- evaluate_checkpoint_with_vid.py 📄
- evaluate_checkpoints.py 📄
- inspect_all_objs.py 📄
</details>

<details>
<summary><strong>models</strong> 🧠</summary>

- **red_ball_grey_models** 📂
  - ppo_redballgrey_baseline.zip 📦
  - ppo_redballgrey_macro.zip 📦
- **red_ball_models** 📂
  - ppo_redball_baseline.zip 📦
  - ppo_redball_macro.zip 📦
</details>

<details>
<summary><strong>snapshots</strong> 📸</summary>

- env_snapshot_20250629_121219.png 📷
- env_snapshot_20250629_131922.png 📷
- output-minigrid.png 📷
</details>

<details>
<summary><strong>training_logs_tensorboard</strong> 📊</summary>

- **CSV_step_data** 📂
  - redball 📄
  - redballgrey 📄
- **redball** 📂
  - redball_baseline 📄
  - redball_macro 📄
- **redballgrey** 📂
  - redballgrey_baseline 📄
  - redballgrey_macro 📄
</details>

<details>
<summary><strong>training_scripts</strong> 💻</summary>

- **red_ball_grey_scripts** 📂
  - macro_wrapper.py 📄
  - raw_state_wrapper.py 📄
  - train_ppo_baseline_rbg.py 📄
  - train_ppo_macro_rbg.py 📄
- **red_ball_scripts** 📂
  - macro_wrapper.py 📄
  - raw_state_wrapper.py 📄
  - train_ppo_baseline_rb.py 📄
  - train_ppo_macro_rb.py 📄
</details>

- requirements.txt 📜

## Directory Explanations 📝

<details>
<summary><strong>evaluation scripts</strong> 📊</summary>

This directory contains tools for evaluating trained agents:
- **initial_evaluators**: Quick evaluators that record short videos, with hard-coded model paths you can edit (redball, redballgrey).
- **plotters**: Scripts to generate figures, including evaluation plot and training plot, from evaluation data.
</details>

<details>
<summary><strong>misc</strong> 🛠️</summary>

A collection of utility scripts used during development and debugging:
- decode_grid.py, decode_grid_with_fig.py: Tools for decoding grid states, with optional figure output.
- envision_symb.py: Visualizes symbolic data.
- evaluate_checkpoint_with_vid.py, evaluate_checkpoints.py: Scripts to assess checkpoints, with video support.
- inspect_all_objs.py: Inspects all objects in the environment.
</details>

<details>
<summary><strong>models</strong> 🧠</summary>

Holds the final trained checkpoints for both tasks and agents:
- **red_ball_grey_models**: ppo_redballgrey_baseline.zip and ppo_redballgrey_macro.zip.
- **red_ball_models**: ppo_redball_baseline.zip and ppo_redball_macro.zip.
</details>

<details>
<summary><strong>training_logs_tensorboard</strong> 📊</summary>

Contains TensorBoard logs for training runs:
- **CSV_step_data**: Per-episode CSVs for redball and redballgrey.
- **redball**: Logs for redball_baseline and redball_macro.
- **redballgrey**: Logs for redballgrey_baseline and redballgrey_macro.
- The plotter scripts in `evaluation scripts/plotters` can turn these logs into figures.
</details>

<details>
<summary><strong>training_scripts</strong> 💻</summary>

Includes scripts used to train the baseline and macro agents for each task:
- **red_ball_grey_scripts**: macro_wrapper.py, raw_state_wrapper.py, train_ppo_baseline_rbg.py, train_ppo_macro_rbg.py.
- **red_ball_scripts**: macro_wrapper.py, raw_state_wrapper.py, train_ppo_baseline_rb.py, train_ppo_macro_rb.py.
</details>

<details>
<summary>requirements.txt 📜</summary>

Lists all Python dependencies required to run the code, ensuring reproducibility.
</details>

---

## Environment

Use Python 3.10+. Required library versions are in `requirements.txt`.

```bash
pip install -r requirements.txt
# If BabyAI envs are not found, install the BabyAI package as well:
pip install "git+https://github.com/mila-iqia/babyai.git"
```

Note: the training scripts set `device="cpu"`. If you have a compatible GPU, you can switch to `"cuda"` inside the script where the PPO model is created. But I advise you use cpu if you are not using `cnnpolicy`during training.

---

## Quick start

### Train: GoToRedBall (baseline)
Run from the task folder so the local imports resolve:

```bash
cd training_scripts/red_ball_scripts
python train_ppo_baseline_rb.py
```

### Train: GoToRedBall (macro version)
```bash
cd training_scripts/red_ball_scripts
python train_ppo_macro_rb.py
```

### Train: GoToRedBallGrey (baseline)
```bash
cd training_scripts/red_ball_grey_scripts
python train_ppo_baseline_rbg.py
```

### Train: GoToRedBallGrey (macro version)
```bash
cd training_scripts/red_ball_grey_scripts
python train_ppo_macro_rbg.py
```

Each script runs for **15,000,000** timesteps with **8** parallel environments, uses **FullyObsWrapper** and a **64-step** time limit, and writes logs and the final model under the corresponding playground folder.

---

## Evaluate

You can evaluate any saved model using the multi-seed evaluator. It writes a CSV per seed and prints summary statistics.

```bash
cd "evaluation scripts/plotters/evaluation plot"

# Example: RedBall, macro agent
python eval_multi_seed.py   --env_id BabyAI-GoToRedBall-v0   --agent macro   --model_path ../../../models/red_ball_models/ppo_redball_macro.zip   --episodes 100   --seeds 0 1 2 3 4   --out_dir out/redball_macro

# Example: RedBall, baseline agent
python eval_multi_seed.py   --env_id BabyAI-GoToRedBall-v0   --agent baseline   --model_path ../../../models/red_ball_models/ppo_redball_baseline.zip   --episodes 100   --seeds 0 1 2 3 4   --out_dir out/redball_baseline

# Example: RedBallGrey, macro agent
python eval_multi_seed.py   --env_id BabyAI-GoToRedBallGrey-v0   --agent macro   --model_path ../../../models/red_ball_grey_models/ppo_redballgrey_macro.zip   --episodes 100   --seeds 0 1 2 3 4   --out_dir out/redballgrey_macro

# Example: RedBallGrey, baseline agent
python eval_multi_seed.py   --env_id BabyAI-GoToRedBallGrey-v0   --agent baseline   --model_path ../../../models/red_ball_grey_models/ppo_redballgrey_baseline.zip   --episodes 100   --seeds 0 1 2 3 4   --out_dir out/redballgrey_baseline
```

To record short videos for a qualitative check, use the initial evaluators and edit the `PPO.load(...)` path inside the script to point to your model zip. Videos are saved under `videos/`.

---

## Macro components

- **Unstuck handler:** detects when the agent’s grid position does not change for a short window and temporarily issues a random action to break the loop. Control returns to the policy on the next step.
- **Reward-tracking (logging only):** stores the type–colour of the most recent rewarded object and logs when a matching object is visible. At the moment, it does not change actions, rewards, or observations. It could serve as a build-up to a promising macro-action development.

Both components are implemented as Gym wrappers and sit after monitoring and before the observation wrapper.

---

## Results at a glance

- GoToRedBall and GoToRedBallGrey were trained for 15M steps with 8 envs.
- The macro agent achieved higher success, higher mean reward, and shorter episodes on average.
- The effect is visible in multi-seed evaluation summaries and in the TensorBoard curves.

Exact numbers, figures, and statistical details are in the dissertation. This code reproduces the setup that produced those results.

---

## Notes and pitfalls

- Run scripts from their folder. Relative imports and log paths assume that working directory.
- If the environment id is not found, install BabyAI as shown above.
- TensorBoard logs are under `training_logs_tensorboard/`; the plotting scripts for the training process expect those folders.
- To use GPU, change `device="cpu"` to `"cuda"` in the training scripts.

Finally, do this in a Linux environment, or WSL2 for Windows. Trying to run everything from Windows might lead you to encounter errors not covered here.

Also, use an MD viewer to view this README file for better visuals.

Have Fun!!!

---

## Acknowledgements

MiniGrid and BabyAI for the environments, and Stable-Baselines3 for the PPO implementation.
- https://minigrid.farama.org/
- https://github.com/mila-iqia/babyai
- https://stable-baselines3.readthedocs.io/en/master/
