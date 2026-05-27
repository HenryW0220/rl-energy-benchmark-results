"""
TD-MPC2 on DMC Cartpole-balance with CodeCarbon energy tracking.
CS 496 RL Energy Benchmarking Project

TD-MPC2 uses a Hydra config system, so we launch train.py as a subprocess
from its package directory (tdmpc2/tdmpc2/).  After training we parse the
eval.csv that Hydra's Logger writes automatically.

Usage:
    python run_tdmpc2.py --seed 0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from codecarbon import EmissionsTracker

TDMPC2_PKG   = Path.home() / "rl_repos" / "tdmpc2" / "tdmpc2"
RESULTS_ROOT = Path("/mnt/d/rl_energy_benchmark/results")

TASK          = "cartpole-balance"
TOTAL_STEPS   = 100_000
EVAL_FREQ     = 10_000
EVAL_EPISODES = 10
MODEL_SIZE    = 5          # smallest preset (~5M params), good for single-task
EXP_NAME      = "energy_benchmark"


def find_eval_csv(seed: int) -> Path:
    """
    TD-MPC2's Logger places eval.csv at:
        <original_cwd>/logs/<task>/<seed>/<exp_name>/eval.csv
    where original_cwd is the directory the subprocess was launched from.
    """
    return TDMPC2_PKG / "logs" / TASK / str(seed) / EXP_NAME / "eval.csv"


def parse_eval_csv(csv_path: Path) -> list[dict]:
    """Read TD-MPC2's eval.csv and return eval_log entries."""
    if not csv_path.exists():
        print(f"  WARNING: eval.csv not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    eval_log = []
    for _, row in df.iterrows():
        eval_log.append({
            "step":        int(row["step"]),
            "mean_reward": float(row["episode_reward"]),
        })
    return eval_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    seed = args.seed

    out_dir = RESULTS_ROOT / "tdmpc2" / "dmc_cartpole" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name = f"tdmpc2_dmc_cartpole_seed{seed}",
        output_dir   = str(out_dir),
        output_file  = "emissions.csv",
        log_level    = "error",
    )

    # Hydra overrides are passed as positional key=value arguments.
    # enable_wandb=false disables W&B (also disables save_video/save_agent).
    # compile=false avoids torch.compile recompilation noise in benchmarks.
    cmd = [
        sys.executable, "train.py",
        f"task={TASK}",
        f"steps={TOTAL_STEPS}",
        f"seed={seed}",
        f"eval_freq={EVAL_FREQ}",
        f"eval_episodes={EVAL_EPISODES}",
        f"model_size={MODEL_SIZE}",
        f"exp_name={EXP_NAME}",
        "enable_wandb=false",
        "save_video=false",
        "compile=false",
    ]

    # TD-MPC2 sets MUJOCO_GL itself (defaults to 'egl'), but make it explicit.
    env_vars = {**os.environ, "MUJOCO_GL": "egl"}

    print(f"\n=== TD-MPC2 | DMC cartpole-balance | seed={seed} | steps={TOTAL_STEPS:,} ===")
    print(f"Output dir : {out_dir}")
    print(f"Command    : {' '.join(cmd)}\n")

    tracker.start()
    t0 = time.time()

    proc = subprocess.run(
        cmd,
        cwd=str(TDMPC2_PKG),
        env=env_vars,
    )

    emissions = tracker.stop()
    elapsed   = time.time() - t0

    if proc.returncode != 0:
        print(f"\nTD-MPC2 exited with return code {proc.returncode} — "
              f"attempting to parse partial results.")

    eval_log = parse_eval_csv(find_eval_csv(seed))

    summary = {
        "model":            "TD-MPC2",
        "env":              "DMC/cartpole-balance",
        "seed":             seed,
        "total_steps":      TOTAL_STEPS,
        "elapsed_seconds":  elapsed,
        "emissions_kg_co2": emissions,
    }

    print(f"\n=== Done | time={elapsed:.0f}s | CO2={emissions:.6f} kg ===")
    print(f"Eval checkpoints : {len(eval_log)}")
    print(f"Results saved to : {out_dir}")

    if proc.returncode != 0:
        # Don't write summary.json on failure so the watchdog won't count this
        # as done and will correctly restart the experiment.
        sys.exit(proc.returncode)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "eval_log.json", "w") as f:
        json.dump(eval_log, f, indent=2)


if __name__ == "__main__":
    main()
