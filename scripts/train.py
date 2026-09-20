"""
Train a MaskablePPO agent (sb3-contrib) on TicTacToeEnv.

- Masked actions: the policy can never sample an occupied cell.
- Trains both as first and second player (randomize_first=True by default).
- Default curriculum: random -> rule_l4 -> mixed (includes minimax).
- Saves model to ``models/ppo_tictactoe.zip`` + eval matrix to ``models/ppo_eval.txt``.

Usage:
    uv run python scripts/train.py --help
    uv run python scripts/train.py --smoke            # quick sanity run (~1 min)
    uv run python scripts/train.py                    # full 1M-step curriculum
"""

import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

from tic_tac_toe_rl.evaluation import eval_matrix, format_matrix, make_thunk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAVE_PATH = os.path.join(REPO_ROOT, "models", "ppo_tictactoe")
DEFAULT_EVAL_PATH = os.path.join(REPO_ROOT, "models", "ppo_eval.txt")


def build_stages(total_timesteps: int, opponent: str | None, no_curriculum: bool):
    if opponent is not None or no_curriculum:
        name = opponent or "random"
        return [(name, total_timesteps)]
    # Default curriculum shares (must sum to 1.0).
    shares = [("random", 0.3), ("rule_l4", 0.3), ("mixed", 0.4)]
    stages = []
    used = 0
    for i, (name, share) in enumerate(shares):
        if i == len(shares) - 1:
            steps = total_timesteps - used
        else:
            steps = int(total_timesteps * share)
            used += steps
        stages.append((name, steps))
    return stages


def train_maskable_ppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    save_path: str = DEFAULT_SAVE_PATH,
    seed: int = 42,
    opponent: str | None = None,
    no_curriculum: bool = False,
    randomize_first: bool = True,
    eval_episodes: int = 300,
    eval_path: str = DEFAULT_EVAL_PATH,
    tensorboard_log: str | None = None,
):
    stages = build_stages(total_timesteps, opponent, no_curriculum)
    print(f"Curriculum: {stages} (randomize_first={randomize_first}, seed={seed})")

    check_env(make_thunk(stages[0][0], randomize_first, seed, 0)(), warn=True)

    model = None
    for i, (opp_name, steps) in enumerate(stages):
        vec_env = DummyVecEnv([make_thunk(opp_name, randomize_first, seed, j) for j in range(n_envs)])
        if model is None:
            model = MaskablePPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                seed=seed,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                n_steps=1024,
                batch_size=512,
                n_epochs=10,
                learning_rate=3e-4,
                ent_coef=0.01,
                tensorboard_log=tensorboard_log,
                device="cpu",
            )
        else:
            model.set_env(vec_env)
        print(f"--- Stage {i + 1}/{len(stages)}: vs {opp_name} for {steps} steps ---")
        model.learn(total_timesteps=steps, reset_num_timesteps=(i == 0))
        vec_env.close()

    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    rows = eval_matrix(model, episodes=eval_episodes, seed=seed)
    summary = format_matrix(rows)
    print(summary)
    os.makedirs(os.path.dirname(eval_path) or ".", exist_ok=True)
    with open(eval_path, "w") as f:
        f.write(summary + "\n")
    return model, rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train MaskablePPO on TicTacToeEnv")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    p.add_argument("--opponent", default=None, help="single opponent, disables curriculum (e.g. random, rule_l4, minimax, mixed)")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--no-randomize-first", action="store_true", help="agent always moves first")
    p.add_argument("--eval-episodes", type=int, default=300)
    p.add_argument("--eval-path", default=DEFAULT_EVAL_PATH)
    p.add_argument("--tensorboard-log", default=None)
    p.add_argument("--smoke", action="store_true", help="20k-step single-opponent sanity run")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        train_maskable_ppo(
            total_timesteps=20_000, n_envs=4, save_path=args.save_path, seed=args.seed,
            opponent="random", no_curriculum=True,
            randomize_first=not args.no_randomize_first, eval_episodes=50,
            eval_path=args.eval_path, tensorboard_log=args.tensorboard_log,
        )
    else:
        train_maskable_ppo(
            total_timesteps=args.timesteps, n_envs=args.n_envs, save_path=args.save_path,
            seed=args.seed, opponent=args.opponent, no_curriculum=args.no_curriculum,
            randomize_first=not args.no_randomize_first, eval_episodes=args.eval_episodes,
            eval_path=args.eval_path, tensorboard_log=args.tensorboard_log,
        )
