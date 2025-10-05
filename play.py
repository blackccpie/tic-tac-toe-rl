"""
Play against a trained PPO agent saved as `ppo_tictactoe.zip`.
- Loads the model and uses deterministic policy.
- You play O (second). Agent is X (first).
- Handles Ctrl-C cleanly and shows a square ASCII board.
"""

import sys
import signal

import numpy as np
from stable_baselines3 import PPO

from tic_tac_toe_env import TicTacToeEnv

INDEX_MAP = "0 1 2\n3 4 5\n6 7 8"

def render_board_ascii(board: np.ndarray, show_indices: bool = False) -> str:
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
        out = out + '|\n\nIndices:\n' + INDEX_MAP
    return out

def human_opponent_policy_factory():
    def human_policy(board: np.ndarray) -> int:
        print('Current board:')
        print(render_board_ascii(board, show_indices=True))

        legal = [int(i) for i in np.flatnonzero(board.ravel() == 0)]
        if not legal:
            print("No legal moves for human. Passing.")
            return None

        while True:
            try:
                raw = input("Your move (0-8 or 1-9), Ctrl-C to quit: ").strip()
            except KeyboardInterrupt:
                raise
            except EOFError:
                raise KeyboardInterrupt

            if raw == "":
                continue
            try:
                val = int(raw)
                if 1 <= val <= 9:
                    val = val - 1
                if val in legal:
                    return int(val)
                else:
                    print("Illegal move (occupied or out of range). Try again.")
            except ValueError:
                print("Invalid input. Enter a number 0..8 or 1..9.")

    return human_policy

def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask a simple yes/no question on the terminal. Returns True for yes.

    default=False means Enter or blank yields No.
    Handles KeyboardInterrupt by re-raising so caller can exit.
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


def play_with_model(model_path: str = "ppo_tictactoe.zip"):
    # Single-line prompt at startup to choose who goes first
    try:
        human_first = prompt_yes_no("Do you want to play first? (y/N):", default=False)
    except KeyboardInterrupt:
        print('Interrupted. Goodbye.')
        sys.exit(0)

    # optional: ask whether to use deterministic actions
    try:
        deterministic = prompt_yes_no("Use deterministic agent actions? (Y/n):", default=True)
    except KeyboardInterrupt:
        print('Interrupted. Goodbye.')
        sys.exit(0)

    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"Could not load model {model_path}: {e}")
        sys.exit(1)

    rng = np.random.default_rng()

    def agent_policy_from_model(board: np.ndarray) -> int:
        obs_vec = board.ravel().astype(np.float32)
        action, _ = model.predict(obs_vec, deterministic=deterministic)
        return int(action)

    human_policy = human_opponent_policy_factory()

    # Decide how to construct the env depending on who goes first
    if human_first:
        # Human will act as the agent (X). The opponent policy should be the trained model.
        env = TicTacToeEnv(opponent_policy=agent_policy_from_model)
    else:
        env = TicTacToeEnv(opponent_policy=human_policy)

    # handle Ctrl-C gracefully
    def _sigint_handler(sig, frame):
        print('Received interrupt. Exiting...')
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            obs, _ = env.reset()
            if human_first:
                print('New game — you play X (first). Agent will play O (second).')
            else:
                print('New game — you play O (second). Agent is X (first).')
            print(render_board_ascii(obs))

            done = False
            while not done:
                if human_first:
                    # Human (as agent X) plays by calling env.step(action)
                    try:
                        a = human_policy(obs)
                    except KeyboardInterrupt:
                        raise
                    if a is None:
                        obs, reward, terminated, truncated, info = obs, 0, False, False, {}
                    else:
                        obs, reward, terminated, truncated, info = env.step(int(a))
                        print(render_board_ascii(obs))

                    if terminated or truncated:
                        if reward == env.win_reward:
                            print("You (X) win!")
                        elif reward == env.loss_reward:
                            print("Agent (O) wins!")
                        else:
                            print("Draw!")
                        done = True
                        break

                    if info.get("opponent_action") is not None:
                        print(f"Agent (O) played: {info['opponent_action']}")
                        print(render_board_ascii(obs))

                else:
                    obs_vec = obs.ravel().astype(np.float32)
                    # sample (or deterministic) according to the earlier prompt
                    action, _ = model.predict(obs_vec, deterministic=deterministic)
                    action = int(action)

                    print(f"Agent (X) plays: {action}")
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(render_board_ascii(obs))

                    if terminated or truncated:
                        if reward == env.win_reward:
                            print("Agent (X) wins!")
                        elif reward == env.loss_reward:
                            print("You (O) win!")
                        else:
                            print("Draw!")
                        done = True
                        break

                    if info.get("opponent_action") is not None:
                        print(f"You (O) played: {info['opponent_action']}")
                        print(render_board_ascii(obs))

            try:
                again = prompt_yes_no("Play again? (y/N):", default=False)
            except KeyboardInterrupt:
                print('Exiting.')
                break
            if not again:
                print("Goodbye")
                break

    except KeyboardInterrupt:
        print('Interrupted by user. Goodbye.')
        sys.exit(0)


if __name__ == "__main__":
    play_with_model()
