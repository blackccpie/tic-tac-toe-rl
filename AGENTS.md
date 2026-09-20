# AGENTS.md

No tests, lint, typecheck, CI, or task runner.

## Layout

- `tic_tac_toe_rl/` — installable package (`uv sync` installs it editable via hatchling; `uv run` resolves imports from anywhere).
  - `tic_tac_toe_env.py` — core `TicTacToeEnv`. Agent mark is always `1`, opponent `2`; move order via `randomize_first` / `reset(options={"first": ...})`.
  - `opponents.py` — opponent pool; build via `get_opponent_policy(name, seed)`. Minimax is `lru_cache`d — keep it.
  - `wrappers.py` — `FlattenAndNormalizeObs` (obs `/2.0` + mask forwarding). Single source of truth for obs encoding.
  - `evaluation.py` — env factories + eval matrix shared by train/eval. Import from here, don't duplicate.
  - `utils.py` — `board_to_obs`, `board_to_mask`, `predict_action`, `load_model` (accepts `models/` and legacy root paths; MaskablePPO→PPO fallback).
- `scripts/` — CLI entrypoints, run as `uv run python scripts/<name>.py` (thin: argparse + calls into the package).
  - `train.py` — MaskablePPO curriculum (`random → rule_l4 → mixed`), writes `models/ppo_tictactoe.zip` + `models/ppo_eval.txt`. Flags: `--timesteps --opponent --no-curriculum --smoke --save-path --eval-path`.
  - `eval.py` — standalone eval matrix (`--model --episodes --seed`).
  - `play.py` (CLI) / `play_gui.py` (needs display) — `--first {ask,human,agent,random} --stochastic`.
- `models/` — generated artifacts (untracked). Retraining overwrites them.

## Setup

- Python 3.12 (`.python-version`, `requires-python = ">=3.12"`).
- `uv sync` then `uv run python scripts/<name>.py`.
- Deps: `gymnasium`, `stable-baselines3`, `sb3-contrib` (MaskablePPO), `pygame`, `numpy`. Force `device="cpu"` — CUDA is broken on this host.

## Gotchas

- Never feed raw `board.ravel()` to a model — always `utils.board_to_obs` (`/2.0`) + `board_to_mask`. `predict_action` does both (with `action_masks` kwarg for MaskablePPO).
- The model trained as X only (own marks read as 1s). When it plays O (human-first mode in `scripts/play*.py`), flip with `utils.flip_board` before predicting — unflipped O-play loses 200/200 to minimax, flipped draws 200/200. Action indices transfer unchanged.
- `ActionMasker` must wrap as `ActionMasker(env, lambda e: e.action_masks())` — the string form `"action_masks"` double-passes `env` and raises TypeError.
- `env.step()` includes the opponent reply; `info` may carry `opponent_action` / `opponent_illegal` / `illegal_move`. Illegal agent move terminates with -1.0; illegal opponent move forfeits (agent wins).
- Eval must cover `agent` / `opponent` / `random` starts vs `random` / `rule_l4` / `minimax` — single random-only eval hides weakness. Illegal count must be 0.
- Full train is ~10 min on CPU (1M steps, 8 envs); use `scripts/train.py --smoke` (20k steps) for code changes.
