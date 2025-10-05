"""
GUI play script using pygame for rendering and clicks.
Save as `play_ppo_sdl.py` next to `tictactoe_sdl.py`.
Install requirements: pip install pygame stable-baselines3 gymnasium
"""

import sys
import signal
import numpy as np
from stable_baselines3 import PPO

from tic_tac_toe_env import TicTacToeEnv
from gui import TicTacToeGUI


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    while True:
        try:
            resp = input(f"{prompt} ")
        except KeyboardInterrupt:
            raise
        except EOFError:
            return default
        if resp == "":
            return default
        if resp.lower().startswith('y'):
            return True
        if resp.lower().startswith('n'):
            return False
        print("Please answer y or n.")


def play_sdl(model_path: str = "ppo_tictactoe.zip"):
    try:
        human_first = prompt_yes_no("Do you want to play first? (y/N):", default=False)
    except KeyboardInterrupt:
        print('Interrupted. Goodbye.')
        sys.exit(0)

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

    gui = TicTacToeGUI("Tic-Tac-Toe (PPO)")

    # human policy that blocks waiting for GUI click
    def human_policy(board: np.ndarray) -> int:
        return gui.wait_for_click(board, prompt="Your move (click an empty square)")

    def agent_policy_from_model(board: np.ndarray) -> int:
        obs_vec = board.ravel().astype(np.float32)
        a, _ = model.predict(obs_vec, deterministic=deterministic)
        return int(a)

    if human_first:
        env = TicTacToeEnv(opponent_policy=agent_policy_from_model)
    else:
        env = TicTacToeEnv(opponent_policy=human_policy)

    # handle ctrl-c from console
    def _sigint_handler(sig, frame):
        print('Received interrupt. Exiting...')
        gui.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            obs, _ = env.reset()
            if human_first:
                gui.show_message('New game — you play X (first).', board=obs)
            else:
                gui.show_message('New game — you play O (second).', board=obs)

            done = False
            while not done:
                if human_first:
                    # human is agent (X) -> they click, then env.step will call agent model as opponent
                    try:
                        a = human_policy(obs)
                    except SystemExit:
                        gui.close()
                        raise
                    except KeyboardInterrupt:
                        gui.close()
                        raise

                    obs, reward, terminated, truncated, info = env.step(int(a))
                    gui.show_message('Move played', board=obs)

                    if terminated or truncated:
                        if reward == env.win_reward:
                            gui.show_message('You (X) win!', board=obs)
                        elif reward == env.loss_reward:
                            gui.show_message('Agent (O) wins!', board=obs)
                        else:
                            gui.show_message('Draw!', board=obs)
                        gui.show_message('Press any key or close window to continue', board=obs)
                        # wait a moment then break
                        gui.wait_for_click(obs, prompt='Click to continue')
                        done = True
                        break

                    # opponent_action present
                    if info.get('opponent_action') is not None:
                        gui.show_message(f"Agent (O) played: {info['opponent_action']}", board=obs)

                else:
                    # agent X plays first
                    obs_vec = obs.ravel().astype(np.float32)
                    action, _ = model.predict(obs_vec, deterministic=deterministic)
                    action = int(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    gui.show_message(f"Agent (X) plays: {action}", board=obs)

                    if terminated or truncated:
                        if reward == env.win_reward:
                            gui.show_message('Agent (X) wins!', board=obs)
                        elif reward == env.loss_reward:
                            gui.show_message('You (O) win!', board=obs)
                        else:
                            gui.show_message('Draw!', board=obs)
                        gui.wait_for_click(obs, prompt='Click to continue')
                        done = True
                        break

                    if info.get('opponent_action') is not None:
                        gui.show_message(f"You (O) played: {info['opponent_action']}", board=obs)

            # ask whether to play again via terminal prompt
            try:
                again = prompt_yes_no("Play again? (y/N):", default=False)
            except KeyboardInterrupt:
                print('Exiting.')
                break
            if not again:
                break

    except KeyboardInterrupt:
        print('Interrupted by user. Goodbye.')
    finally:
        gui.close()


if __name__ == "__main__":
    play_sdl()
