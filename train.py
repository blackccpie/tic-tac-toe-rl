"""
Train a PPO agent (stable-baselines3) on the TicTacToeEnv.

Requirements:
    pip install stable-baselines3 gymnasium

Notes:
- Uses a small MLP policy. Observations are flattened and normalized to [0, 1].
- Trains against the environment's default opponent (random).
- Saves model to `ppo_tictactoe.zip` and a sample evaluation summary `ppo_eval.txt`.
"""

import os
from typing import Callable

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

from tic_tac_toe_env import TicTacToeEnv

class FlattenAndNormalizeObs(gym.ObservationWrapper):
    """Flatten 3x3 int board to 9-d float32 vector, normalized to [0, 1].
    
    Board values: 0 (empty), 1 (agent/X), 2 (opponent/O)
    Normalized: 0.0, 0.5, 1.0
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

    def observation(self, observation):
        # Normalize [0, 1, 2] -> [0.0, 0.5, 1.0]
        return (observation.ravel() / 2.0).astype(np.float32)

def make_wrapped_env() -> Callable[[], gym.Env]:
    def _thunk():
        env = TicTacToeEnv()
        env = FlattenAndNormalizeObs(env)
        return env

    return _thunk

def evaluate_model(model: PPO, env: gym.Env, episodes: int = 500):
    """
    Evaluate a trained model against the (possibly wrapped) env.
    Uses env.unwrapped (when available) to read reward constants like win_reward.
    
    Note: obs is already flattened and normalized by FlattenAndNormalizeObs wrapper.
    """
    base_env = getattr(env, "unwrapped", env)  # unwrap if wrapped
    wins = draws = losses = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # obs is already flattened float32 from the wrapper
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        # compare final reward to the base environment's reward constants
        if reward == getattr(base_env, "win_reward", 1.0):
            wins += 1
        elif reward == getattr(base_env, "loss_reward", -1.0):
            losses += 1
        else:
            draws += 1

    return wins, draws, losses

def train_ppo(
    total_timesteps: int = 200_000,
    n_envs: int = 8,
    save_path: str = "ppo_tictactoe",
    seed: int = 0,
):
    # Create vectorized envs
    vec_env = DummyVecEnv([make_wrapped_env() for _ in range(n_envs)])

    # optional: check environment compatibility with SB3
    check_env(make_wrapped_env()(), warn=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        batch_size=256,
        device="cpu"
    )

    model.learn(total_timesteps=total_timesteps)

    # save
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    # evaluation against single env (non-vectorized)
    eval_env = make_wrapped_env()()
    wins, draws, losses = evaluate_model(model, eval_env, episodes=1000)
    summary = f"Eval over 1000 games -> W/D/L: {wins}/{draws}/{losses}"

    print(summary)
    with open("ppo_eval.txt", "w") as f:
        f.write(summary)

    vec_env.close()

if __name__ == "__main__":
    # small default run; increase total_timesteps for better performance
    train_ppo(total_timesteps=1000_000, n_envs=8, save_path="ppo_tictactoe", seed=42)

