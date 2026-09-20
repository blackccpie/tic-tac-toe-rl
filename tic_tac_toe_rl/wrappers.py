"""Gymnasium observation wrappers for Tic-Tac-Toe."""

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class FlattenAndNormalizeObs(gym.ObservationWrapper):
    """Flatten 3x3 int board to 9-d float32 vector, normalized to [0, 1].

    Board values: 0 (empty), 1 (agent/X), 2 (opponent/O)
    Normalized: 0.0, 0.5, 1.0
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)

    def observation(self, observation):
        return (observation.ravel() / 2.0).astype(np.float32)

    def action_masks(self):
        base = self.env
        while hasattr(base, "env") and not hasattr(base, "board"):
            base = base.env
        if hasattr(base, "action_masks"):
            return np.asarray(base.action_masks(), dtype=bool)
        return np.ones(9, dtype=bool)
