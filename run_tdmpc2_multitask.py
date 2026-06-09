"""
TD-MPC2 multi-task on DMC cartpole-balance + walker-walk - 100k steps each task
Usage: python3 run_tdmpc2_multitask.py --seed 0

Requires: ~/rl_repos/tdmpc2 repo, tdmpc2 conda/venv with all deps
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from codecarbon import EmissionsTracker

# Add tdmpc2 repo to path
TDMPC2_ROOT = os.path.expanduser("~/rl_repos/tdmpc2")
sys.path.insert(0, TDMPC2_ROOT)
sys.path.insert(0, os.path.join(TDMPC2_ROOT, "tdmpc2"))

TOTAL_STEPS_PER_TASK = 100_000
EVAL_FREQ            = 10_000
EVAL_EPISODES        = 10
RESULTS_ROOT         = "/mnt/d/rl_energy_benchmark/results"

TASKS = ["cartpole-balance", "walker-walk"]


def make_env(task_name):
    """Create a DMC env using TD-MPC2's own env factory if available, else dm_control directly."""
    try:
        from envs.dmcontrol import make_env as tdmpc_make
        return tdmpc_make(task_name)
    except Exception:
        from dm_control import suite
        import gymnasium as gym
        from gymnasium import spaces

        domain, task = task_name.split("-", 1)
        task = task.replace("-", "_")

        class DMCWrapper(gym.Env):
            def __init__(self):
                self._env = suite.load(domain, task)
                obs_spec = self._env.observation_spec()
                obs_size = sum(int(np.prod(v.shape)) for v in obs_spec.values())
                self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
                act_spec = self._env.action_spec()
                self.action_space = spaces.Box(
                    act_spec.minimum.astype(np.float32),
                    act_spec.maximum.astype(np.float32), dtype=np.float32)

            def reset(self, **kwargs):
                ts = self._env.reset()
                return self._obs(ts), {}

            def step(self, action):
                ts = self._env.step(action)
                obs = self._obs(ts)
                r = float(ts.reward) if ts.reward is not None else 0.0
                return obs, r, ts.last(), False, {}

            def _obs(self, ts):
                return np.concatenate([v.flatten() for v in ts.observation.values()]).astype(np.float32)

        return DMCWrapper()


def evaluate(agent, envs, n_episodes=10):
    """Run n_episodes on each task env, return dict of mean rewards."""
    results = {}
    for task_name, env in envs.items():
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset() if hasattr(env, 'reset') else (env.reset(), {})
            done = False
            ep_r = 0.0
            while not done:
                with __import__('torch').no_grad():
                    action = agent.act(obs, t0=False, eval_mode=True)
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()
                result = env.step(action)
                obs, r, term, trunc = result[0], result[1], result[2], result[3]
                ep_r += r
                done = term or trunc
            rewards.append(ep_r)
        results[task_name] = float(np.mean(rewards))
    return results


def main(seed):
    out_dir = os.path.join(RESULTS_ROOT, f"TDMPC2_multitask_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== TD-MPC2 multi-task | tasks={TASKS} | seed={seed} ===")

    tracker = EmissionsTracker(
        output_dir=out_dir,
        project_name=f"TDMPC2_multitask_seed{seed}",
        log_level="error",
    )

    # ── Try to use TD-MPC2's native training loop ─────────────────────────────
    # TD-MPC2 uses hydra, so we invoke its train.py directly as a subprocess
    # with a custom config that enables multi-task mode.
    import subprocess

    config_overrides = [
        f"seed={seed}",
        f"steps={TOTAL_STEPS_PER_TASK * len(TASKS)}",
        "task=mt_cartpole_walker",   # we'll write a custom task list config below
        "multitask=true",
        f"tasks=[cartpole-balance,walker-walk]",
        "obs=state",
        "device=cuda",
        "save_video=false",
        "wandb=false",
        f"exp_name=multitask_seed{seed}",
        f"work_dir={out_dir}/tdmpc2_run",
    ]

    train_script = os.path.join(TDMPC2_ROOT, "tdmpc2", "train.py")

    # Check if train.py supports multitask via CLI; if not, fall back to
    # running two sequential single-task runs and reporting combined energy.
    if not os.path.exists(train_script):
        train_script = os.path.join(TDMPC2_ROOT, "train.py")

    print(f"Using train script: {train_script}")
    print("Starting CodeCarbon tracker...")

    tracker.start()
    start_time = datetime.now()

    # Try multi-task mode first
    mt_success = False
    try:
        cmd = [sys.executable, train_script] + config_overrides
        print(f"Running: {' '.join(cmd[:6])} ...")
        result = subprocess.run(cmd, capture_output=False, timeout=7200,
                                cwd=os.path.join(TDMPC2_ROOT, "tdmpc2"))
        if result.returncode == 0:
            mt_success = True
            print("Multi-task run completed successfully.")
        else:
            print(f"Multi-task run failed (returncode={result.returncode}), falling back to sequential.")
    except Exception as e:
        print(f"Multi-task launch failed: {e}, falling back to sequential single-task runs.")

    if not mt_success:
        # Fallback: run cartpole then walker sequentially under the same tracker
        print("\nFallback: running cartpole-balance then walker-walk sequentially.")
        for task in TASKS:
            task_cfg = [
                f"seed={seed}",
                f"steps={TOTAL_STEPS_PER_TASK}",
                f"task={task}",
                "obs=state",
                "model_size=5",
                "enable_wandb=false",
                "save_video=false",
                "compile=false",
                f"exp_name=sequential_{task}_seed{seed}",
            ]
            cmd = [sys.executable, train_script] + task_cfg
            print(f"\n  Running {task}: {' '.join(cmd[:5])} ...")
            subprocess.run(cmd, cwd=os.path.join(TDMPC2_ROOT, "tdmpc2"))

    emissions = tracker.stop()
    elapsed   = (datetime.now() - start_time).total_seconds()

    summary = {
        "model": "TDMPC2",
        "mode": "multitask" if mt_success else "sequential_singletask",
        "tasks": TASKS,
        "steps_per_task": TOTAL_STEPS_PER_TASK,
        "total_steps": TOTAL_STEPS_PER_TASK * len(TASKS),
        "seed": seed,
        "elapsed_seconds": elapsed,
        "emissions_kg_co2": emissions,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Done | mode={'multitask' if mt_success else 'sequential'} | "
          f"time={elapsed:.0f}s | CO2={emissions:.6f} kg ===")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
