"""
Play against a trained agent saved as `models/ppo_tictactoe.zip`.

Usage:
    uv run python scripts/play.py
    uv run python scripts/play.py --first human     # you are X (first)
    uv run python scripts/play.py --first agent     # you are O (second)
    uv run python scripts/play.py --first random --stochastic
"""

import argparse
import os
import sys
import signal

import numpy as np

from tic_tac_toe_rl.tic_tac_toe_env import TicTacToeEnv
from tic_tac_toe_rl.utils import flip_board, load_model, predict_action, prompt_yes_no, render_board_ascii

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(REPO_ROOT, "models", "ppo_tictactoe.zip")


def human_opponent_policy_factory():
    def human_policy(board: np.ndarray) -> int:
        print("Current board:")
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Play CLI tic-tac-toe vs trained model")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--first", choices=["ask", "human", "agent", "random"], default="ask")
    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    return p.parse_args(argv)


def play_with_model(model_path: str = DEFAULT_MODEL, first: str = "ask",
                    deterministic: bool = True):
    if first == "ask":
        try:
            human_first = prompt_yes_no("Do you want to play first? (y/N):", default=False)
        except KeyboardInterrupt:
            print("Interrupted. Goodbye.")
            sys.exit(0)
        try:
            deterministic = prompt_yes_no(
                "Use deterministic agent actions? (Y/n):", default=deterministic
            )
        except KeyboardInterrupt:
            print("Interrupted. Goodbye.")
            sys.exit(0)
    elif first == "random":
        human_first = bool(np.random.random() < 0.5)
    else:
        human_first = first == "human"

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Could not load model {model_path}: {e}")
        sys.exit(1)

    def agent_policy_from_model(board: np.ndarray) -> int:
        # When the human plays first, the model is O: flip marks so its own
        # pieces read as 1s, matching the X-perspective it trained on.
        if human_first:
            board = flip_board(board)
        return predict_action(model, board, deterministic=deterministic)

    human_policy = human_opponent_policy_factory()

    if human_first:
        env = TicTacToeEnv(opponent_policy=agent_policy_from_model)
    else:
        env = TicTacToeEnv(opponent_policy=human_policy)

    def _sigint_handler(sig, frame):
        print("Received interrupt. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            obs, _ = env.reset()
            if human_first:
                print("New game — you play X (first). Agent will play O (second).")
            else:
                print("New game — you play O (second). Agent is X (first).")
            print(render_board_ascii(obs))

            done = False
            while not done:
                if human_first:
                    try:
                        a = human_policy(obs)
                    except KeyboardInterrupt:
                        raise
                    if a is None:
                        obs, reward, terminated, truncated, info = obs, 0, False, False, {}
                    else:
                        obs, reward, terminated, truncated, info = env.step(int(a))

                    if terminated or truncated:
                        if reward == env.win_reward:
                            print(render_board_ascii(obs))
                            print("You (X) win!")
                        elif reward == env.loss_reward:
                            print(render_board_ascii(obs))
                            print("Agent (O) wins!")
                        else:
                            print(render_board_ascii(obs))
                            print("Draw!")
                        done = True
                        break

                    if info.get("opponent_action") is not None:
                        print(f"Agent (O) played: {info['opponent_action']}")
                    print(render_board_ascii(obs))

                else:
                    action = predict_action(model, obs, deterministic=deterministic)
                    print(f"Agent (X) plays: {action}")
                    obs, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        if reward == env.win_reward:
                            print(render_board_ascii(obs))
                            print("Agent (X) wins!")
                        elif reward == env.loss_reward:
                            print(render_board_ascii(obs))
                            print("You (O) win!")
                        else:
                            print(render_board_ascii(obs))
                            print("Draw!")
                        done = True
                        break

                    # obs already includes the human reply move; print once.
                    if info.get("opponent_action") is not None:
                        print(f"You (O) played: {info['opponent_action']}")
                    print(render_board_ascii(obs))

            try:
                again = prompt_yes_no("Play again? (y/N):", default=False)
            except KeyboardInterrupt:
                print("Exiting.")
                break
            if not again:
                print("Goodbye")
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    play_with_model(model_path=args.model, first=args.first, deterministic=args.deterministic)
