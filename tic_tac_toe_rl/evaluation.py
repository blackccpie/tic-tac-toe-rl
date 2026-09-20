"""Shared env factories + eval matrix vs the opponent pool."""

import numpy as np

from sb3_contrib.common.wrappers import ActionMasker

from tic_tac_toe_rl.tic_tac_toe_env import TicTacToeEnv
from tic_tac_toe_rl.wrappers import FlattenAndNormalizeObs
from tic_tac_toe_rl.opponents import get_opponent_policy

OPPONENTS = ["random", "rule_l4", "minimax"]
FIRST_MODES = ["agent", "opponent", "random"]


def _masked(env):
    return ActionMasker(env, lambda e: e.action_masks())


def make_thunk(opponent_name: str, randomize_first: bool, seed: int, idx: int):
    def _thunk():
        opp = get_opponent_policy(opponent_name, seed=seed + idx)
        env = TicTacToeEnv(opponent_policy=opp, randomize_first=randomize_first)
        env.reset(seed=seed + idx)
        return _masked(FlattenAndNormalizeObs(env))

    return _thunk


def make_single_env(opponent_name: str, randomize_first: bool, seed: int):
    opp = get_opponent_policy(opponent_name, seed=seed)
    env = TicTacToeEnv(opponent_policy=opp, randomize_first=randomize_first)
    env.reset(seed=seed)
    return _masked(FlattenAndNormalizeObs(env))


def _base_env(env):
    base = env.unwrapped
    while hasattr(base, "env") and not hasattr(base, "win_reward"):
        base = base.env
    return base


def evaluate_vs(model, opponent_name: str, episodes: int, first: str, seed: int = 0):
    """Evaluate model vs one opponent. first: 'agent' | 'opponent' | 'random'."""
    env = make_single_env(opponent_name, randomize_first=False, seed=seed)
    base = _base_env(env)
    wins = draws = losses = illegals = 0
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        first_this = first if first in ("agent", "opponent") else ("opponent" if rng.random() < 0.5 else "agent")
        obs, _ = env.reset(options={"first": first_this})
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        if info.get("illegal_move"):
            illegals += 1
            losses += 1
        elif reward == getattr(base, "win_reward", 1.0):
            wins += 1
        elif reward == getattr(base, "loss_reward", -1.0):
            losses += 1
        else:
            draws += 1
    env.close()
    return {"W": wins, "D": draws, "L": losses, "illegal": illegals}


def eval_matrix(model, episodes: int, seed: int = 0):
    rows = {}
    for opp in OPPONENTS:
        for mode in FIRST_MODES:
            rows[(opp, mode)] = evaluate_vs(model, opp, episodes, mode, seed=seed)
    return rows


def format_matrix(rows):
    lines = ["Eval matrix (W/D/L, illegal moves) — agent-first / opponent-first / mixed:"]
    for opp in OPPONENTS:
        cells = []
        for mode in FIRST_MODES:
            r = rows[(opp, mode)]
            cells.append(f"{mode}={r['W']}/{r['D']}/{r['L']} ill={r['illegal']}")
        lines.append(f"  vs {opp:10s} " + " | ".join(cells))
    return "\n".join(lines)
