"""
Autonomous watchdog for RL energy benchmark experiments.

Monitors the active experiment, detects stalls (no metrics update for
STALL_THRESHOLD seconds), kills and restarts the stuck experiment, then
lets run_all_remaining.sh continue with the next ones.

Logic:
  - The main run is bash run_all_remaining.sh (sequential, set -e).
  - We don't restart the whole shell script; instead we detect stalls and
    kill the stuck DreamerV3 / TD-MPC2 subprocess, which causes the parent
    run_dreamerv3.py / run_tdmpc2.py to exit non-zero, which causes
    run_all_remaining.sh to stop (set -euo pipefail).
  - We then clean the partial logdir for that seed and restart the whole
    run_all_remaining.sh, which skips already-completed experiments via
    the summary.json check added below.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BENCH          = Path("/mnt/d/rl_energy_benchmark")
RESULTS        = BENCH / "results"
LOG            = BENCH / "run_all_remaining.log"
WATCHDOG_LOG   = BENCH / "watchdog.log"
STALL_SECONDS  = 10 * 60   # 10 min without a metrics update = stall
POLL_INTERVAL  = 60         # check every 60 s

DREAMER_EXPS = [
    ("dreamerv3", "atari_pong",   0),
    ("dreamerv3", "atari_pong",   1),
    ("dreamerv3", "atari_pong",   2),
    ("dreamerv3", "dmc_cartpole", 0),
    ("dreamerv3", "dmc_cartpole", 1),
    ("dreamerv3", "dmc_cartpole", 2),
]
TDMPC2_EXPS = [
    ("tdmpc2", "dmc_cartpole", 0),
    ("tdmpc2", "dmc_cartpole", 1),
    ("tdmpc2", "dmc_cartpole", 2),
]
ALL_EXPS = DREAMER_EXPS + TDMPC2_EXPS


def wlog(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[watchdog {ts}] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")


def summary_exists(model, env, seed):
    return (RESULTS / model / env / f"seed_{seed}" / "summary.json").exists()


def all_done():
    return all(summary_exists(m, e, s) for m, e, s in ALL_EXPS)


def count_done():
    return sum(1 for m, e, s in ALL_EXPS if summary_exists(m, e, s))


def metrics_path(model, env, seed):
    if model == "dreamerv3":
        return RESULTS / model / env / f"seed_{seed}" / "dreamer_logdir" / "metrics.jsonl"
    else:  # tdmpc2
        # find the eval.csv inside logs/<task>/<seed>/energy_benchmark/
        task = "dmc_cartpole_balance"
        pkg = Path.home() / "rl_repos" / "tdmpc2" / "tdmpc2"
        return pkg / "logs" / task / str(seed) / "energy_benchmark" / "eval.csv"


def get_current_experiment():
    """Return the first experiment that lacks summary.json."""
    for m, e, s in ALL_EXPS:
        if not summary_exists(m, e, s):
            return m, e, s
    return None


def get_runner_pids():
    """Return PIDs of run_all_remaining.sh, run_dreamerv3.py, run_tdmpc2.py, main.py, train.py."""
    patterns = [
        "run_all_remaining.sh",
        "run_dreamerv3.py",
        "run_tdmpc2.py",
        "dreamerv3/main.py",
        "tdmpc2/train.py",
    ]
    pids = []
    try:
        out = subprocess.check_output(["pgrep", "-f", "run_all_remaining.sh"], text=True).strip()
        pids += [int(p) for p in out.split() if p]
    except subprocess.CalledProcessError:
        pass
    for pat in patterns[1:]:
        try:
            out = subprocess.check_output(["pgrep", "-f", pat], text=True).strip()
            pids += [int(p) for p in out.split() if p]
        except subprocess.CalledProcessError:
            pass
    return list(set(pids))


def kill_all_runners():
    pids = get_runner_pids()
    if not pids:
        return
    wlog(f"Killing PIDs: {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(2)


def clean_partial(model, env, seed):
    """Remove partial logdir so the experiment restarts cleanly."""
    if model == "dreamerv3":
        logdir = RESULTS / model / env / f"seed_{seed}" / "dreamer_logdir"
        if logdir.exists():
            wlog(f"Removing partial logdir: {logdir}")
            subprocess.run(["rm", "-rf", str(logdir)], check=True)
            logdir.mkdir(parents=True, exist_ok=True)
        # remove partial outputs
        for f in ["summary.json", "eval_log.json", "emissions.csv"]:
            p = RESULTS / model / env / f"seed_{seed}" / f
            if p.exists():
                p.unlink()


def launch_runner():
    wlog("Launching run_all_remaining.sh in background")
    truncate_log()
    proc = subprocess.Popen(
        ["bash", "run_all_remaining.sh"],
        cwd=str(BENCH),
        stdout=open(LOG, "a"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    wlog(f"Runner PID: {proc.pid}")
    return proc


def truncate_log():
    with open(LOG, "a"):
        pass  # ensure exists


def runner_alive():
    return bool(get_runner_pids())


def main():
    wlog("=== Watchdog started ===")
    wlog(f"Experiments done at start: {count_done()}/9")

    if all_done():
        wlog("All 9 experiments already complete. Exiting.")
        return

    # If nothing is running, start it.
    if not runner_alive():
        wlog("No runner found — launching.")
        launch_runner()

    last_metrics_mtime = {}    # (model, env, seed) -> float

    def update_mtime(m, e, s):
        p = metrics_path(m, e, s)
        if p.exists():
            last_metrics_mtime[(m, e, s)] = p.stat().st_mtime

    while not all_done():
        time.sleep(POLL_INTERVAL)

        done = count_done()
        wlog(f"Progress: {done}/9 done. Runner alive: {runner_alive()}")

        if all_done():
            break

        cur = get_current_experiment()
        if cur is None:
            break

        m, e, s = cur

        # Check if runner is alive at all
        if not runner_alive():
            wlog(f"Runner died before completing {m}/{e}/seed_{s}. Restarting.")
            clean_partial(m, e, s)
            launch_runner()
            time.sleep(30)
            continue

        # Check for stall on current experiment
        p = metrics_path(m, e, s)
        if not p.exists():
            # Experiment hasn't written anything yet — it may still be
            # compiling/initialising. Give it extra time.
            update_mtime(m, e, s)
            wlog(f"  {m}/{e}/seed_{s}: no metrics file yet (still initialising)")
            continue

        current_mtime = p.stat().st_mtime
        prev_mtime = last_metrics_mtime.get((m, e, s), 0)

        if prev_mtime == 0:
            # First time we've seen this file — record and move on
            last_metrics_mtime[(m, e, s)] = current_mtime
            wlog(f"  {m}/{e}/seed_{s}: metrics file first seen")
            continue

        age = time.time() - current_mtime
        wlog(f"  {m}/{e}/seed_{s}: metrics age {age:.0f}s (stall threshold {STALL_SECONDS}s)")

        if age > STALL_SECONDS:
            wlog(f"STALL DETECTED on {m}/{e}/seed_{s} — killing and restarting")
            kill_all_runners()
            clean_partial(m, e, s)
            time.sleep(5)
            launch_runner()
            # reset mtime tracking for this experiment
            del last_metrics_mtime[(m, e, s)]
            time.sleep(60)  # give new runner time to boot
        else:
            last_metrics_mtime[(m, e, s)] = current_mtime

    wlog(f"=== All {count_done()}/9 experiments complete ===")
    # Print final summary
    for m, e, s in ALL_EXPS:
        p = RESULTS / m / e / f"seed_{s}" / "summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            wlog(f"  {m}/{e}/seed_{s}: {d.get('elapsed_seconds', '?'):.0f}s, "
                 f"CO2={d.get('emissions_kg_co2', '?'):.6f} kg")
        else:
            wlog(f"  {m}/{e}/seed_{s}: MISSING summary.json")


if __name__ == "__main__":
    main()
