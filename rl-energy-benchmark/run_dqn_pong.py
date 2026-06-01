"""
DQN on Atari Pong - 2M env steps
CE 496 RL Energy Benchmarking Project
Usage: python3 run_dqn_pong.py --seed 0
"""

import argparse
import os
import json
import numpy as np
from datetime import datetime
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
from codecarbon import EmissionsTracker

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_STEPS   = 2_000_000
EVAL_FREQ     = 100_000
EVAL_EPISODES = 10
RESULTS_ROOT  = "results"

# ── Callback ──────────────────────────────────────────────────────────────────
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, log_path):
        super().__init__()
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path        = log_path
        self.results         = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = [False]
                ep_reward = 0
                while not done[0]:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    ep_reward += reward[0]
                rewards.append(ep_reward)
            mean_r = float(np.mean(rewards))
            self.results.append({"step": self.num_timesteps, "mean_reward": mean_r})
            print(f"  [Eval] step={self.num_timesteps:>8}  mean_reward={mean_r:.2f}")
            with open(self.log_path, "w") as f:
                json.dump(self.results, f, indent=2)
        return True


def main(seed):
    out_dir = os.path.join(RESULTS_ROOT, "dqn", "pong", f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    train_env = VecFrameStack(make_atari_env("ALE/Pong-v5", n_envs=1, seed=seed), n_stack=4)
    eval_env  = VecFrameStack(make_atari_env("ALE/Pong-v5", n_envs=1, seed=seed+100), n_stack=4)

    eval_cb = EvalCallback(
        eval_env        = eval_env,
        eval_freq       = EVAL_FREQ,
        n_eval_episodes = EVAL_EPISODES,
        log_path        = os.path.join(out_dir, "eval_log.json"),
    )

    tracker = EmissionsTracker(
        project_name = f"dqn_pong_seed{seed}",
        output_dir   = out_dir,
        output_file  = "emissions.csv",
        log_level    = "error",
    )

    model = DQN(
        "CnnPolicy",
        train_env,
        learning_rate          = 1e-4,
        buffer_size            = 100_000,
        learning_starts        = 1_000,
        batch_size             = 32,
        gamma                  = 0.99,
        train_freq             = 4,
        target_update_interval = 1_000,
        exploration_fraction   = 0.1,
        seed                   = seed,
        verbose                = 0,
        device                 = "cuda",
    )

    print(f"\n=== DQN | Pong | seed={seed} | steps={TOTAL_STEPS:,} ===")
    print(f"Output dir: {out_dir}\n")

    tracker.start()
    start_time = datetime.now()
    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb)
    emissions = tracker.stop()
    elapsed   = (datetime.now() - start_time).total_seconds()

    summary = {
        "model": "DQN",
        "env": "ALE/Pong-v5",
        "seed": seed,
        "total_steps": TOTAL_STEPS,
        "elapsed_seconds": elapsed,
        "emissions_kg_co2": emissions,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    model.save(os.path.join(out_dir, "model"))
    train_env.close()
    eval_env.close()

    print(f"\n=== Done | time={elapsed:.0f}s | CO2={emissions:.6f} kg ===")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
