"""
DreamerV3 with CodeCarbon energy tracking.
CS 496 RL Energy Benchmarking Project

Runs DreamerV3 as a subprocess from its repo directory (dreamerv3/main.py),
wraps the full run with CodeCarbon, then parses scores.jsonl to produce
eval_log.json bucketed at every 10k env steps.

NOTE on eval strategy: DreamerV3's train_eval mode runs evaluation episodes
sequentially (debug=True by default), which stalls at ~6 fps and makes 10-episode
eval rounds take ~28 minutes each — incompatible with report_every=60s.
We therefore use --script train and derive performance from training episode
scores in scores.jsonl, bucketed at 10k-step intervals.  This is a valid and
common approach for MBRL benchmarks where training scores track learning progress.

Usage:
    python run_dreamerv3.py --env atari_pong   --seed 0
    python run_dreamerv3.py --env dmc_cartpole --seed 0
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from codecarbon import EmissionsTracker

DREAMER_REPO  = Path.home() / "rl_repos" / "dreamerv3"
RESULTS_ROOT  = Path("/mnt/d/rl_energy_benchmark/results")
EVAL_FREQ     = 10_000   # bucket width in env steps

ENV_CONFIGS = {
    "atari_pong": {
        # size12m ≈ 12M params (vs 165M default) → ~35 min/seed at full train_ratio.
        # The 165M default at train_ratio=256 gives ~6 fps and 267 min/seed on 4070 Ti.
        # IMPORTANT: pass as a list so subprocess sees separate argv elements.
        "configs":      ["atari100k", "size12m"],
        "task":         "atari100k_pong",
        "total_steps":  100_000,
        "log_every":    20,
        "model_label":  "DreamerV3",
        "env_label":    "atari100k/Pong",
    },
    "dmc_cartpole": {
        "configs":      ["dmc_proprio"],
        "task":         "dmc_cartpole_balance",
        "total_steps":  100_000,
        "log_every":    20,
        "model_label":  "DreamerV3",
        "env_label":    "DMC/cartpole-balance",
        # run.debug=True + run.envs=1: disables portal subprocess spawning which
        # causes malloc heap corruption (JAX CUDA init + MuJoCo EGL in subprocess).
        # Sequential single-env mode is slower (~20 fps) but produces correct results.
        "extra_flags":  ["--run.debug", "True", "--run.envs", "1"],
    },
}


def parse_scores_jsonl(logdir: Path) -> list[dict]:
    """
    Read scores.jsonl and return eval_log entries bucketed at every EVAL_FREQ steps.

    In --script train mode, scores.jsonl has one line per completed training episode:
        {"step": <global_env_step>, "episode/score": <return>}

    We group by the lower 10k boundary and average all episodes in each bucket,
    giving a mean return at approximately every 10k-step checkpoint.
    """
    scores_file = logdir / "scores.jsonl"
    if not scores_file.exists():
        print(f"  WARNING: scores.jsonl not found at {scores_file}")
        return []

    buckets: dict[int, list[float]] = collections.defaultdict(list)
    with open(scores_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "episode/score" not in d or "step" not in d:
                continue
            step = int(d["step"])
            score = float(d["episode/score"])
            bucket = (step // EVAL_FREQ) * EVAL_FREQ   # floor to nearest 10k
            if bucket == 0:
                bucket = EVAL_FREQ                      # skip pre-training episodes
            buckets[bucket].append(score)

    eval_log = []
    for bucket_step in sorted(buckets):
        scores = buckets[bucket_step]
        eval_log.append({
            "step": bucket_step,
            "mean_reward": float(sum(scores) / len(scores)),
            "num_episodes": len(scores),
        })
    return eval_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",  choices=list(ENV_CONFIGS), required=True,
                        help="atari_pong or dmc_cartpole")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg  = ENV_CONFIGS[args.env]
    seed = args.seed

    out_dir = RESULTS_ROOT / "dreamerv3" / args.env / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DreamerV3 writes its own logs here
    logdir = out_dir / "dreamer_logdir"
    logdir.mkdir(exist_ok=True)

    tracker = EmissionsTracker(
        project_name = f"dreamerv3_{args.env}_seed{seed}",
        output_dir   = str(out_dir),
        output_file  = "emissions.csv",
        log_level    = "error",
    )

    cmd = [
        sys.executable,
        str(DREAMER_REPO / "dreamerv3" / "main.py"),
        "--configs",           *cfg["configs"],
        "--task",              cfg["task"],
        "--seed",              str(seed),
        "--logdir",            str(logdir),
        "--script",            "train",           # pure training; no stalling eval driver
        f"--run.steps={cfg['total_steps']}",
        f"--run.log_every={cfg['log_every']}",
        "--run.save_every=3600",
        "--jax.profiler=False",                    # disable: CUPTI stop_trace deadlocks on WSL2
        *cfg.get("extra_flags", []),
    ]

    # Carry the full environment but make sure the repo is on PYTHONPATH
    env_vars = {**os.environ, "PYTHONPATH": str(DREAMER_REPO)}

    print(f"\n=== DreamerV3 | {args.env} | seed={seed} | steps={cfg['total_steps']:,} ===")
    print(f"Output dir : {out_dir}")
    print(f"Command    : {' '.join(cmd)}\n")

    tracker.start()
    t0 = time.time()

    proc = subprocess.run(
        cmd,
        cwd=str(DREAMER_REPO),
        env=env_vars,
    )

    emissions = tracker.stop()
    elapsed   = time.time() - t0

    if proc.returncode != 0:
        print(f"\nDreamerV3 exited with return code {proc.returncode} — "
              f"attempting to parse partial results.")

    eval_log = parse_scores_jsonl(logdir)

    summary = {
        "model":            cfg["model_label"],
        "env":              cfg["env_label"],
        "seed":             seed,
        "total_steps":      cfg["total_steps"],
        "elapsed_seconds":  elapsed,
        "emissions_kg_co2": emissions,
    }

    print(f"\n=== Done | time={elapsed:.0f}s | CO2={emissions:.6f} kg ===")
    print(f"Eval checkpoints : {len(eval_log)}")
    print(f"Results saved to : {out_dir}")

    if proc.returncode != 0:
        sys.exit(proc.returncode)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "eval_log.json", "w") as f:
        json.dump(eval_log, f, indent=2)


if __name__ == "__main__":
    main()
