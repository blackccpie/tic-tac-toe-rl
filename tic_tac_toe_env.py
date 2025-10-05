"""
TicTacToe Gymnasium Environment with optional reward shaping.
- Base rewards: win, loss, draw, illegal move, step penalty.
- Optional shaping: small bonus for attack (creating two-in-a-row with empty third),
  or defense (blocking opponent's immediate win).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TicTacToeEnv(gym.Env):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self,
                 win_reward=1.0,
                 loss_reward=-1.0,
                 draw_reward=0.5,
                 illegal_move_reward=-1.0,
                 step_penalty=-0.01,
                 attack_reward=0.1,
                 defense_reward=0.1,
                 opponent_policy=None):
        super().__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=int)

        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        self.illegal_move_reward = illegal_move_reward
        self.step_penalty = step_penalty
        self.attack_reward = attack_reward
        self.defense_reward = defense_reward

        self.opponent_policy = opponent_policy if opponent_policy is not None else self._random_policy

        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 = agent, 2 = opponent

        # Winning lines (all rows, columns, diagonals)
        self.winning_lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]

    def _random_policy(self, board):
        legal = [i for i in range(9) if board.ravel()[i] == 0]
        return np.random.choice(legal) if legal else None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board[:] = 0
        self.current_player = 1
        return self.board.copy(), {}

    def step(self, action):
        r, c = divmod(action, 3)
        done = False
        reward = self.step_penalty

        if self.board[r, c] != 0:
            # Illegal move
            return self.board.copy(), self.illegal_move_reward, True, False, {"illegal_move": True}

        # Place agent move
        self.board[r, c] = 1

        # Reward shaping: attack/defense detection
        reward += self._compute_shaping(action, player=1)

        if self._check_winner(1):
            return self.board.copy(), self.win_reward, True, False, {}

        if not (self.board == 0).any():
            return self.board.copy(), self.draw_reward, True, False, {}

        # Opponent move
        opp_action = self.opponent_policy(self.board.copy())
        if opp_action is None:
            return self.board.copy(), self.draw_reward, True, False, {}
        or_, oc = divmod(opp_action, 3)
        self.board[or_, oc] = 2

        if self._check_winner(2):
            return self.board.copy(), self.loss_reward, True, False, {"opponent_action": opp_action}

        if not (self.board == 0).any():
            return self.board.copy(), self.draw_reward, True, False, {"opponent_action": opp_action}

        return self.board.copy(), reward, False, False, {"opponent_action": opp_action}

    def render(self):
        chars = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(chars[x] for x in row) for row in self.board]
        return "\n".join(rows)

    def close(self):
        pass

    def _check_winner(self, player):
        for line in self.winning_lines:
            if all(self.board[r, c] == player for r, c in line):
                return True
        return False

    def _compute_shaping(self, action, player):
        extra = 0.0
        r, c = divmod(action, 3)

        # Attack: does this move now make a 2-in-a-row with empty third?
        for line in self.winning_lines:
            marks = [self.board[r_, c_] for r_, c_ in line]
            if marks.count(player) == 2 and marks.count(0) == 1:
                if (r, c) in line:
                    extra += self.attack_reward
                    break

        # Defense: did this move block opponent’s imminent win?
        opponent = 2 if player == 1 else 1
        # Check opponent winning chances before the move
        board_before = self.board.copy()
        board_before[r, c] = 0
        imminent = []
        for line in self.winning_lines:
            marks = [board_before[r_, c_] for r_, c_ in line]
            if marks.count(opponent) == 2 and marks.count(0) == 1:
                imminent.append(line)
        if imminent:
            # If our action is in one of those imminent lines, we blocked
            for line in imminent:
                if (r, c) in line:
                    extra += self.defense_reward
                    break

        return extra
