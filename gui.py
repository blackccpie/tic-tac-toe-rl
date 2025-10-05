"""
Minimal SDL-based GUI for Tic-Tac-Toe using pygame (SDL wrapper).
Provides `TicTacToeGUI` class which can display a board and block waiting for a human click.

Save as `tictactoe_sdl.py` next to other scripts.
Requires: `pip install pygame`
"""

import pygame
import sys
import numpy as np

CELL_SIZE = 120
GRID_SIZE = 3
MARGIN = 10
WINDOW_W = CELL_SIZE * GRID_SIZE + MARGIN * 2
WINDOW_H = CELL_SIZE * GRID_SIZE + MARGIN * 2 + 40  # extra for status line

class TicTacToeGUI:
    def __init__(self, caption: str = "Tic-Tac-Toe"):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(caption)
        self.font = pygame.font.SysFont(None, 28)
        self.small_font = pygame.font.SysFont(None, 20)
        self.clock = pygame.time.Clock()
        self.status = ""

    def draw_board(self, board: np.ndarray):
        self.screen.fill((240, 240, 240))

        # draw grid and marks
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + r * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, (255, 255, 255), rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)

                v = int(board[r, c])
                if v == 1:
                    # draw X
                    pygame.draw.line(self.screen, (200, 30, 30), (x + 16, y + 16), (x + CELL_SIZE - 16, y + CELL_SIZE - 16), 6)
                    pygame.draw.line(self.screen, (200, 30, 30), (x + CELL_SIZE - 16, y + 16), (x + 16, y + CELL_SIZE - 16), 6)
                elif v == 2:
                    # draw O
                    center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
                    pygame.draw.circle(self.screen, (30, 30, 200), center, CELL_SIZE // 2 - 16, 6)

        # status line
        status_surf = self.small_font.render(self.status, True, (10, 10, 10))
        self.screen.blit(status_surf, (MARGIN, WINDOW_H - 32))

        pygame.display.flip()

    def pos_to_action(self, pos):
        mx, my = pos
        # ignore clicks outside the grid
        if mx < MARGIN or my < MARGIN or mx > MARGIN + GRID_SIZE * CELL_SIZE or my > MARGIN + GRID_SIZE * CELL_SIZE:
            return None
        c = (mx - MARGIN) // CELL_SIZE
        r = (my - MARGIN) // CELL_SIZE
        if 0 <= r < 3 and 0 <= c < 3:
            return int(r * 3 + c)
        return None

    def wait_for_click(self, board: np.ndarray, prompt: str = "Your move"):
        """Block until the user clicks on an empty cell. Update the display while waiting."""
        self.status = prompt
        self.draw_board(board)
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit(0)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    action = self.pos_to_action(ev.pos)
                    if action is None:
                        continue
                    r, c = divmod(action, 3)
                    if board[r, c] == 0:
                        return action
            # keep UI responsive
            self.clock.tick(30)

    def show_message(self, msg: str, board: np.ndarray = None, pause: float = 0.0):
        self.status = msg
        if board is not None:
            self.draw_board(board)
        else:
            # draw only status
            self.screen.fill((240, 240, 240))
            status_surf = self.font.render(msg, True, (10, 10, 10))
            self.screen.blit(status_surf, (MARGIN, WINDOW_H // 2 - 10))
            pygame.display.flip()
        if pause > 0:
            pygame.time.delay(int(pause * 1000))

    def close(self):
        pygame.quit()
