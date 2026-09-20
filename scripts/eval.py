"""Evaluate a trained model against an opponent matrix.

Usage:
    uv run python scripts/eval.py --help
    uv run python scripts/eval.py                       # eval models/ppo_tictactoe.zip, 300 games/cell
    uv run python scripts/eval.py --model models/ppo_tictactoe.zip --episodes 100
"""

import argparse
import os

from tic_tac_toe_rl.utils import load_model
from tic_tac_toe_rl.evaluation import eval_matrix, format_matrix

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ppo_tictactoe.zip"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Eval tic-tac-toe model vs opponent matrix")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = load_model(args.model)
    rows = eval_matrix(model, episodes=args.episodes, seed=args.seed)
    print(format_matrix(rows))


if __name__ == "__main__":
    main()
