"""
TicTacToe Gymnasium Environment with optional reward shaping.

- Base rewards: win, loss, draw, illegal move, step penalty.
- Optional shaping: small bonus for attack (creating two-in-a-row with empty third),
  or defense (blocking opponent's immediate win).
- Agent mark is always 1 (X); opponent mark is always 2 (O). Move order is
  controlled by ``randomize_first`` / reset ``options`` so the agent can train
  both as first and second player.
- Exposes ``action_masks()`` for MaskablePPO (sb3-contrib).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        win_reward=1.0,
        loss_reward=-1.0,
        draw_reward=0.5,
        illegal_move_reward=-1.0,
        step_penalty=-0.01,
        attack_reward=0.1,
        defense_reward=0.1,
        opponent_policy=None,
        randomize_first=False,
    ):
        super().__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.int64)

        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        self.illegal_move_reward = illegal_move_reward
        self.step_penalty = step_penalty
        self.attack_reward = attack_reward
        self.defense_reward = defense_reward

        self.opponent_policy = opponent_policy if opponent_policy is not None else self._random_policy
        self.randomize_first = randomize_first

        self.board = np.zeros((3, 3), dtype=np.int64)

        # Winning lines (all rows, columns, diagonals)
        self.winning_lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]

    # -- helpers ---------------------------------------------------------
    def _random_policy(self, board):
        legal = [i for i in range(9) if board.ravel()[i] == 0]
        if not legal:
            return None
        rng = getattr(self, "np_random", None)
        if rng is not None:
            return int(rng.choice(legal))
        return int(np.random.choice(legal))

    def action_masks(self):
        """Boolean mask of legal actions, for MaskablePPO."""
        return (self.board.ravel() == 0)

    def _check_winner(self, player):
        for line in self.winning_lines:
            if all(self.board[r, c] == player for r, c in line):
                return True
        return False

    def _place_opponent_opening(self):
        """Let the opponent move first (used when agent trains as 2nd player)."""
        opp_action = self.opponent_policy(self.board.copy())
        if opp_action is None:
            return
        if not isinstance(opp_action, (int, np.integer)):
            return
        opp_action = int(opp_action)
        if 0 <= opp_action < 9:
            r, c = divmod(opp_action, 3)
            if self.board[r, c] == 0:
                self.board[r, c] = 2

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board[:] = 0

        first = "agent"
        if options is not None and "first" in options:
            first = options["first"]
        elif self.randomize_first:
            first = "opponent" if self.np_random.random() < 0.5 else "agent"

        opponent_started = False
        if first == "opponent":
            self._place_opponent_opening()
            opponent_started = True

        return self.board.copy(), {"opponent_started": opponent_started}

    def step(self, action):
        try:
            action = int(action)
        except (TypeError, ValueError):
            return self.board.copy(), self.illegal_move_reward, True, False, {"illegal_move": True}
        if action < 0 or action >= 9:
            return self.board.copy(), self.illegal_move_reward, True, False, {"illegal_move": True}

        r, c = divmod(action, 3)
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
        try:
            opp_action = int(opp_action)
        except (TypeError, ValueError):
            return self.board.copy(), self.win_reward, True, False, {"opponent_illegal": True}
        if opp_action < 0 or opp_action >= 9:
            return self.board.copy(), self.win_reward, True, False, {"opponent_illegal": True}
        or_, oc = divmod(opp_action, 3)
        if self.board[or_, oc] != 0:
            # Opponent forfeits on illegal move; credit the agent with a win.
            return self.board.copy(), self.win_reward, True, False, {"opponent_illegal": True}
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

        # Defense: did this move block opponent's imminent win?
        opponent = 2 if player == 1 else 1
        # Temporarily remove our move to see board state before we played
        board_before = self.board.copy()
        board_before[r, c] = 0

        # Find all lines where opponent had 2 marks and 1 empty (their winning threat)
        for line in self.winning_lines:
            marks = [board_before[r_, c_] for r_, c_ in line]
            if marks.count(opponent) == 2 and marks.count(0) == 1:
                # Find the empty position in this line
                empty_pos = None
                for (lr, lc) in line:
                    if board_before[lr, lc] == 0:
                        empty_pos = (lr, lc)
                        break
                # If our move was placed at that empty position, we blocked it
                if empty_pos == (r, c):
                    extra += self.defense_reward
                    break

        return extra
