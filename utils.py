"""
Shared utilities for the Tic-Tac-Toe RL project.
"""

import sys


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
