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
    # Ask who plays first
    try:
        human_first = prompt_yes_no("Do you want to play first? (y/N):", default=False)
    except KeyboardInterrupt:
        print('\nInterrupted. Goodbye.')
        sys.exit(0)

    # Ask whether to use deterministic actions
    try:
        deterministic = prompt_yes_no("Use deterministic agent actions? (Y/n):", default=True)
    except KeyboardInterrupt:
        print('\nInterrupted. Goodbye.')
        sys.exit(0)

    # Load trained PPO model
    try:
        model = PPO.load(model_path, device="cpu")
    except Exception as e:
        print(f"Could not load model {model_path}: {e}")
        sys.exit(1)

    gui = TicTacToeGUI("Tic-Tac-Toe (PPO)")

    # Human plays via GUI click
    def human_policy(board: np.ndarray) -> int:
        return gui.wait_for_click(board, prompt="Your move (click an empty square)")

    # Agent policy using PPO
    def agent_policy_from_model(board: np.ndarray) -> int:
        obs_vec = board.ravel().astype(np.float32)
        a, _ = model.predict(obs_vec, deterministic=deterministic)
        return int(a)

    # Choose environment opponent depending on who plays first
    if human_first:
        env = TicTacToeEnv(opponent_policy=agent_policy_from_model)
    else:
        env = TicTacToeEnv(opponent_policy=human_policy)

    # Handle Ctrl-C globally
    def _sigint_handler(sig, frame):
        print('\nReceived interrupt. Exiting...')
        gui.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while True:
            obs, _ = env.reset()
            
            # === Splash before the game starts ===
            if human_first:
                gui.wait_for_end(obs, 'New game — you are X (first)')
            else:
                gui.wait_for_end(obs, 'New game — you are O (second)')

            done = False
            while not done:
                if human_first:
                    # Human turn
                    a = human_policy(obs)
                    obs, reward, terminated, truncated, info = env.step(int(a))
                    gui.draw_board(obs)

                    if terminated or truncated:
                        if reward == env.win_reward:
                            msg = 'You (X) win!'
                        elif reward == env.loss_reward:
                            msg = 'Agent (O) wins!'
                        else:
                            msg = 'Draw!'
                        gui.wait_for_end(obs, msg)
                        done = True
                        break

                else:
                    # Agent turn
                    obs_vec = obs.ravel().astype(np.float32)
                    action, _ = model.predict(obs_vec, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    gui.draw_board(obs)

                    if terminated or truncated:
                        if reward == env.win_reward:
                            msg = 'Agent (X) wins!'
                        elif reward == env.loss_reward:
                            msg = 'You (O) win!'
                        else:
                            msg = 'Draw!'
                        gui.wait_for_end(obs, msg)
                        done = True
                        break

            # After game ends, loop continues for new game
            # User can quit with ESC or window close in wait_for_end

    except KeyboardInterrupt:
        print('\nInterrupted by user. Goodbye.')
    finally:
        gui.close()


if __name__ == "__main__":
    play_sdl()
