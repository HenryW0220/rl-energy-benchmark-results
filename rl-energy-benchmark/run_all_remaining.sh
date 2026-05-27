#!/usr/bin/env bash
# Run all 9 benchmark experiments sequentially.
# Idempotent: skips any experiment whose summary.json already exists.
set -euo pipefail

BENCH=/mnt/d/rl_energy_benchmark
RESULTS=$BENCH/results
LOG=$BENCH/run_all_remaining.log
exec > >(tee -a "$LOG") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_exp() {
    local label=$1; local summary_path=$2; shift 2
    if [[ -f "$summary_path" ]]; then
        echo "  SKIP (already done): $label"
        return 0
    fi
    echo ""
    echo "========================================================"
    echo "  START: $label  ($(ts))"
    echo "========================================================"
    local rc=0
    python3 "$@" || rc=$?
    echo "  FINISH: $label  rc=$rc  ($(ts))"
    echo "========================================================"
    return $rc
}

cd "$BENCH"

# ── DreamerV3 · Atari Pong (size12m ≈ 12M params, ~35 min/seed) ──────────────
run_exp "DreamerV3 atari_pong seed=0" "$RESULTS/dreamerv3/atari_pong/seed_0/summary.json" \
    run_dreamerv3.py --env atari_pong --seed 0
run_exp "DreamerV3 atari_pong seed=1" "$RESULTS/dreamerv3/atari_pong/seed_1/summary.json" \
    run_dreamerv3.py --env atari_pong --seed 1
run_exp "DreamerV3 atari_pong seed=2" "$RESULTS/dreamerv3/atari_pong/seed_2/summary.json" \
    run_dreamerv3.py --env atari_pong --seed 2

# ── DreamerV3 · DMC Cartpole-balance (size1m ≈ 1M params, ~5 min/seed) ───────
run_exp "DreamerV3 dmc_cartpole seed=0" "$RESULTS/dreamerv3/dmc_cartpole/seed_0/summary.json" \
    run_dreamerv3.py --env dmc_cartpole --seed 0
run_exp "DreamerV3 dmc_cartpole seed=1" "$RESULTS/dreamerv3/dmc_cartpole/seed_1/summary.json" \
    run_dreamerv3.py --env dmc_cartpole --seed 1
run_exp "DreamerV3 dmc_cartpole seed=2" "$RESULTS/dreamerv3/dmc_cartpole/seed_2/summary.json" \
    run_dreamerv3.py --env dmc_cartpole --seed 2

# ── TD-MPC2 · DMC Cartpole-balance (~30-60 min/seed) ─────────────────────────
run_exp "TD-MPC2 dmc_cartpole seed=0" "$RESULTS/tdmpc2/dmc_cartpole/seed_0/summary.json" \
    run_tdmpc2.py --seed 0
run_exp "TD-MPC2 dmc_cartpole seed=1" "$RESULTS/tdmpc2/dmc_cartpole/seed_1/summary.json" \
    run_tdmpc2.py --seed 1
run_exp "TD-MPC2 dmc_cartpole seed=2" "$RESULTS/tdmpc2/dmc_cartpole/seed_2/summary.json" \
    run_tdmpc2.py --seed 2

echo ""
echo "========================================================"
echo "  ALL 9 EXPERIMENTS COMPLETE  ($(ts))"
echo "========================================================"
