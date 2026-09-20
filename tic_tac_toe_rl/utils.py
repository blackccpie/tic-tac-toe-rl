"""
Shared utilities for the Tic-Tac-Toe RL project.
"""

import numpy as np


def board_to_obs(board) -> np.ndarray:
    """Flatten 3x3 int board (0/1/2) to normalized float32 vector (0.0/0.5/1.0).

    Must match ``FlattenAndNormalizeObs`` in tic_tac_toe_rl/wrappers.py. All
    inference code (play, eval, self-play) must use this — never raw ``board.ravel()``.
    """
    return (np.asarray(board).ravel() / 2.0).astype(np.float32)


def board_to_mask(board) -> np.ndarray:
    """Boolean legality mask for MaskablePPO, shape (9,)."""
    return (np.asarray(board).ravel() == 0)


def flip_board(board) -> np.ndarray:
    """Swap marks 1<->2 so a model trained as X can play as O.

    The model only ever saw own marks as 1 (0.5) and opponent marks as 2
    (1.0). When it plays O, its marks are 2s — flip before predicting.
    Square indices are unaffected, so the predicted action transfers directly.
    """
    b = np.asarray(board).copy()
    m1 = b == 1
    m2 = b == 2
    b[m1] = 2
    b[m2] = 1
    return b


def load_model(model_path: str):
    """Load a MaskablePPO model, falling back to PPO for legacy checkpoints.

    Accepts ``models/ppo_tictactoe.zip`` (current) and bare
    ``ppo_tictactoe.zip`` (legacy root location) interchangeably.
    """
    import os

    candidates = [model_path]
    base = os.path.basename(model_path)
    if model_path.startswith("models/"):
        candidates.append(base)  # legacy root location
    else:
        candidates.append(os.path.join("models", base))

    from sb3_contrib import MaskablePPO

    last_exc = None
    for cand in candidates:
        try:
            return MaskablePPO.load(cand, device="cpu")
        except Exception as e:
            last_exc = e
    try:
        from stable_baselines3 import PPO

        for cand in candidates:
            try:
                return PPO.load(cand, device="cpu")
            except Exception as e:
                last_exc = e
    except ImportError as e:
        last_exc = e
    raise RuntimeError(f"Could not load model {model_path}: {last_exc}")


def predict_action(model, board, deterministic: bool = True) -> int:
    """Predict one action from a raw (3,3) board with correct obs + masking."""
    obs_vec = board_to_obs(board)
    try:
        action, _ = model.predict(
            obs_vec, deterministic=deterministic, action_masks=board_to_mask(board)
        )
    except TypeError:
        # Legacy PPO model without mask support.
        action, _ = model.predict(obs_vec, deterministic=deterministic)
    return int(action)


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask a simple yes/no question on the terminal. Returns True for yes.
    
    Args:
        prompt: The question to display.
        default: Default answer if user presses Enter without typing.
        
    Returns:
        True if user answers 'y' or 'Y', False if 'n' or 'N', 
        or the default if Enter is pressed.
        
    Raises:
        KeyboardInterrupt: If user presses Ctrl-C.
    """
    while True:
        try:
            resp = input(f"{prompt} ")
        except KeyboardInterrupt:
            raise
        except EOFError:
            # treat EOF as No / cancel
            return default
        if resp == "":
            return default
        if resp.lower().startswith('y'):
            return True
        if resp.lower().startswith('n'):
            return False
        print("Please answer y or n.")


def render_board_ascii(board, show_indices: bool = False) -> str:
    """Render a 3x3 board as ASCII art.
    
    Args:
        board: numpy array of shape (3, 3) with values 0 (empty), 1 (X), 2 (O).
        show_indices: If True, show position indices below the board.
        
    Returns:
        String representation of the board.
    """
    import numpy as np
    chars = {0: '.', 1: 'X', 2: 'O'}
    lines = []
    sep = '+---+---+---+'
    lines.append(sep)
    for r in range(3):
        row = board[r]
        lines.append('| ' + ' | '.join(chars[int(x)] for x in row) + ' |')
        lines.append(sep)

    out = '\n'.join(lines)
    if show_indices:
        out = out + '\n\nIndices:\n0 1 2\n3 4 5\n6 7 8'
    return out
