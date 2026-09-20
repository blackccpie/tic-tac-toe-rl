"""
Opponent policies for Tic-Tac-Toe.

Contract: every policy is ``fn(board) -> int | None`` where ``board`` is a
``(3, 3)`` numpy array copy with values 0 (empty), 1 (agent/X), 2 (opponent/O),
and it is the opponent's (O, player 2) turn. Return a legal action 0-8, or
None if the board is full.

Use :func:`get_opponent_policy` to build a seeded policy by name (used by
training curriculum and eval matrix).
"""

import numpy as np

WINNING_LINES = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]

CENTER = 4
CORNERS = [0, 2, 6, 8]


def get_legal_moves(board):
    return [int(i) for i in range(9) if board.ravel()[i] == 0]


def _check_winner_board(board, player):
    for line in WINNING_LINES:
        if all(board[r, c] == player for r, c in line):
            return True
    return False


def _find_winning_move(board, player):
    """Return an action completing 3-in-a-row for player, or None."""
    for line in WINNING_LINES:
        marks = [board[r, c] for r, c in line]
        if marks.count(player) == 2 and marks.count(0) == 1:
            for (r, c) in line:
                if board[r, c] == 0:
                    return int(r * 3 + c)
    return None


def random_policy(board):
    legal = get_legal_moves(board)
    if not legal:
        return None
    return int(np.random.choice(legal))


def rule_based_policy(board, level=4):
    """Level 1: random. L2: center/corners bias. L3: +block. L4: +win."""
    legal = get_legal_moves(board)
    if not legal:
        return None
    if level <= 1:
        return int(np.random.choice(legal))
    if level >= 4:
        win = _find_winning_move(board, player=2)
        if win is not None:
            return win
    if level >= 3:
        block = _find_winning_move(board, player=1)
        if block is not None:
            return block
    # Positional bias: center, then corners, then random.
    if CENTER in legal and (level == 2 or np.random.random() < 0.6):
        return CENTER
    free_corners = [c for c in CORNERS if c in legal]
    if free_corners and np.random.random() < 0.7:
        return int(np.random.choice(free_corners))
    return int(np.random.choice(legal))


def _minimax(board, player_to_move):
    """Return (score, action) from player 2's perspective: +1 O wins, -1 X wins."""
    from functools import lru_cache

    key = (tuple(int(x) for x in board.ravel()), player_to_move)
    return _minimax_cached(key)


from functools import lru_cache


@lru_cache(maxsize=20000)
def _minimax_cached(key):
    flat, player_to_move = key
    board = [[flat[r * 3 + c] for c in range(3)] for r in range(3)]

    def winner(p):
        for line in WINNING_LINES:
            if all(board[r][c] == p for r, c in line):
                return True
        return False

    if winner(2):
        return 1, None
    if winner(1):
        return -1, None
    legal = [i for i, v in enumerate(flat) if v == 0]
    if not legal:
        return 0, None
    if player_to_move == 2:
        best_score, best_action = -2, legal[0]
        for a in legal:
            lst = list(flat)
            lst[a] = 2
            score, _ = _minimax_cached((tuple(lst), 1))
            if score > best_score:
                best_score, best_action = score, a
                if best_score == 1:
                    break
        return best_score, best_action
    else:
        best_score, best_action = 2, legal[0]
        for a in legal:
            lst = list(flat)
            lst[a] = 1
            score, _ = _minimax_cached((tuple(lst), 2))
            if score < best_score:
                best_score, best_action = score, a
                if best_score == -1:
                    break
        return best_score, best_action


def minimax_policy(board):
    legal = get_legal_moves(board)
    if not legal:
        return None
    _, action = _minimax(board.copy(), player_to_move=2)
    if action is None:
        return int(np.random.choice(legal))
    return int(action)


def minimax_randomized_policy(board, random_prob=0.15):
    legal = get_legal_moves(board)
    if not legal:
        return None
    if np.random.random() < random_prob:
        return int(np.random.choice(legal))
    return minimax_policy(board)


def mixed_policy(board):
    """Blend of minimax / rule-L4 / random; used as final curriculum stage."""
    legal = get_legal_moves(board)
    if not legal:
        return None
    roll = np.random.random()
    if roll < 0.6:
        return minimax_policy(board)
    if roll < 0.9:
        return rule_based_policy(board, level=4)
    return int(np.random.choice(legal))


def get_opponent_policy(name="random", level=4, random_prob=0.15, seed=None):
    """Build a seeded opponent policy closure by name.

    Names: 'random', 'rulebased' (use level), 'rule_l1'..'rule_l4',
    'minimax', 'minimax_random', 'mixed'.
    """
    rng = np.random.default_rng(seed)

    def _choice(legal):
        return int(rng.choice(legal))

    def random_seeded(board):
        legal = get_legal_moves(board)
        return _choice(legal) if legal else None

    def rule_seeded(board, lv=level):
        legal = get_legal_moves(board)
        if not legal:
            return None
        if lv <= 1:
            return _choice(legal)
        if lv >= 4:
            win = _find_winning_move(board, player=2)
            if win is not None:
                return win
        if lv >= 3:
            block = _find_winning_move(board, player=1)
            if block is not None:
                return block
        if CENTER in legal and (lv == 2 or rng.random() < 0.6):
            return CENTER
        free_corners = [c for c in CORNERS if c in legal]
        if free_corners and rng.random() < 0.7:
            return int(rng.choice(free_corners))
        return _choice(legal)

    def minimax_rand_seeded(board):
        legal = get_legal_moves(board)
        if not legal:
            return None
        if rng.random() < random_prob:
            return _choice(legal)
        return minimax_policy(board)

    def mixed_seeded(board):
        legal = get_legal_moves(board)
        if not legal:
            return None
        roll = rng.random()
        if roll < 0.6:
            return minimax_policy(board)
        if roll < 0.9:
            return rule_seeded(board, lv=4)
        return _choice(legal)

    key = name.lower()
    if key == "random":
        return random_seeded
    if key in ("rulebased", "rule", f"rule_l{level}", f"rulebased_l{level}"):
        return rule_seeded
    if key in ("rule_l1", "rule_l2", "rule_l3", "rule_l4"):
        lv = int(key[-1])
        return lambda b: rule_seeded(b, lv=lv)
    if key == "minimax":
        return minimax_policy
    if key == "minimax_random":
        return minimax_rand_seeded
    if key == "mixed":
        return mixed_seeded
    raise ValueError(f"Unknown opponent policy: {name!r}")
