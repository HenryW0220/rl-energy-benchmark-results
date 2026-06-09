"""
PPO on DMC walker-walk - 100k env steps
Usage: python3 run_ppo_walker.py --seed 0
"""

import argparse
import os
import json
import numpy as np
from datetime import datetime
from codecarbon import EmissionsTracker

import gym
from gym import spaces
from dm_control import suite
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

TOTAL_STEPS   = 100_000
EVAL_FREQ     = 10_000
EVAL_EPISODES = 10
RESULTS_ROOT  = "/mnt/d/rl_energy_benchmark/results"


class DMCWrapper(gym.Env):
    def __init__(self, domain, task):
        self._env = suite.load(domain, task)
        obs_spec = self._env.observation_spec()
        obs_size = sum(int(np.prod(v.shape)) for v in obs_spec.values())
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        act_spec = self._env.action_spec()
        self.action_space = spaces.Box(act_spec.minimum.astype(np.float32),
                                       act_spec.maximum.astype(np.float32), dtype=np.float32)

    def reset(self):
        ts = self._env.reset()
        return self._get_obs(ts)

    def step(self, action):
        ts = self._env.step(action)
        obs = self._get_obs(ts)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        done = ts.last()
        return obs, reward, done, {}

    def _get_obs(self, ts):
        return np.concatenate([v.flatten() for v in ts.observation.values()]).astype(np.float32)


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, out_dir):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.out_dir = out_dir
        self.log = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                ep_r = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, done, _ = self.eval_env.step(action)
                    ep_r += r
                rewards.append(ep_r)
            mean_r = float(np.mean(rewards))
            self.log.append({"step": self.n_calls, "mean_reward": mean_r})
            print(f"  [eval] step={self.n_calls} mean_reward={mean_r:.2f}")
        return True


def main(seed):
    out_dir = os.path.join(RESULTS_ROOT, f"PPO_dmc_walker_walk_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    def make_env():
        return DMCWrapper("walker", "walk")

    train_env = DummyVecEnv([make_env])
    eval_env  = make_env()

    tracker = EmissionsTracker(
        output_dir=out_dir,
        project_name=f"PPO_walker_walk_seed{seed}",
        log_level="error",
    )

    eval_cb = EvalCallback(eval_env, EVAL_FREQ, EVAL_EPISODES, out_dir)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.1,
        seed=seed,
        verbose=0,
        device="cuda",
    )

    print(f"\n=== PPO | walker-walk | seed={seed} | steps={TOTAL_STEPS:,} ===")
    tracker.start()
    start_time = datetime.now()

    model.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb)

    emissions = tracker.stop()
    elapsed   = (datetime.now() - start_time).total_seconds()

    summary = {
        "model": "PPO",
        "env": "dmc_walker_walk",
        "seed": seed,
        "total_steps": TOTAL_STEPS,
        "elapsed_seconds": elapsed,
        "emissions_kg_co2": emissions,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "eval_log.json"), "w") as f:
        json.dump(eval_cb.log, f, indent=2)

    model.save(os.path.join(out_dir, "model"))
    train_env.close()
    eval_env.close()
    print(f"=== Done | time={elapsed:.0f}s | CO2={emissions:.6f} kg ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
