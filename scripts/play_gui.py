"""
GUI play script using pygame for rendering and clicks.

Usage:
    uv run python scripts/play_gui.py
    uv run python scripts/play_gui.py --first human
    uv run python scripts/play_gui.py --first agent --stochastic
Requires a display; exits with an error message when headless.
"""

import argparse
import os
import sys
import signal
import numpy as np

from tic_tac_toe_rl.tic_tac_toe_env import TicTacToeEnv
from tic_tac_toe_rl.gui import TicTacToeGUI
from tic_tac_toe_rl.utils import flip_board, load_model, predict_action, prompt_yes_no

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(REPO_ROOT, "models", "ppo_tictactoe.zip")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Play GUI tic-tac-toe vs trained model")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--first", choices=["ask", "human", "agent", "random"], default="ask")
    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    return p.parse_args(argv)


def play_sdl(model_path: str = DEFAULT_MODEL, first: str = "ask",
             deterministic: bool = True):
    if first == "ask":
        try:
            human_first = prompt_yes_no("Do you want to play first? (y/N):", default=False)
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye.")
            sys.exit(0)
        try:
            deterministic = prompt_yes_no(
                "Use deterministic agent actions? (Y/n):", default=deterministic
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye.")
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

    try:
        gui = TicTacToeGUI("Tic-Tac-Toe (PPO)")
    except Exception as e:
        print(f"Could not open GUI (need a display): {e}")
        sys.exit(1)

    def human_policy(board: np.ndarray) -> int:
        return gui.wait_for_click(board, prompt="Your move (click an empty square)")

    def agent_policy_from_model(board: np.ndarray) -> int:
        # When the human plays first, the model is O: flip marks so its own
        # pieces read as 1s, matching the X-perspective it trained on.
        if human_first:
            board = flip_board(board)
        return predict_action(model, board, deterministic=deterministic)

    if human_first:
        env = TicTacToeEnv(opponent_policy=agent_policy_from_model)
    else:
        env = TicTacToeEnv(opponent_policy=human_policy)

    def _sigint_handler(sig, frame):
        print("\nReceived interrupt. Exiting...")
        gui.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            obs, _ = env.reset()

            if human_first:
                gui.wait_for_end(obs, "New game — you are X (first)")
            else:
                gui.wait_for_end(obs, "New game — you are O (second)")

            done = False
            while not done:
                if human_first:
                    a = human_policy(obs)
                    obs, reward, terminated, truncated, info = env.step(int(a))
                    gui.draw_board(obs)

                    if terminated or truncated:
                        if reward == env.win_reward:
                            msg = "You (X) win!"
                        elif reward == env.loss_reward:
                            msg = "Agent (O) wins!"
                        else:
                            msg = "Draw!"
                        gui.wait_for_end(obs, msg)
                        done = True
                        break

                else:
                    action = predict_action(model, obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    gui.draw_board(obs)

                    if terminated or truncated:
                        if reward == env.win_reward:
                            msg = "Agent (X) wins!"
                        elif reward == env.loss_reward:
                            msg = "You (O) win!"
                        else:
                            msg = "Draw!"
                        gui.wait_for_end(obs, msg)
                        done = True
                        break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye.")
    finally:
        gui.close()


if __name__ == "__main__":
    args = parse_args()
    play_sdl(model_path=args.model, first=args.first, deterministic=args.deterministic)
