# Tic-Tac-Toe Reinforcement Learning

A reinforcement learning project for training a Tic-Tac-Toe playing agent using PPO (Proximal Policy Optimization) from Stable Baselines3.

## Features

- **Gymnasium Environment**: Custom `TicTacToeEnv` with reward shaping (attack/defense bonuses)
- **MaskablePPO Training**: sb3-contrib action masking — the policy can never sample an occupied cell
- **Multiple Interfaces**: Play against trained agents via CLI or Pygame GUI
- **Opponent Policies**: `opponents.py` (random → rule-based L1-L4 → minimax) with curriculum learning
- **Both sides**: trains as first AND second player (`randomize_first=True`)
- **Normalized Observations**: `utils.board_to_obs` scales inputs to [0, 1]; all inference must use it

## Project Structure

```
tic-tac-toe-rl/
├── tic_tac_toe_rl/       # installable package (uv sync installs it editable)
│   ├── tic_tac_toe_env.py  # Gymnasium environment (core, masked, both-sides)
│   ├── opponents.py        # Opponent pool (random, rule-based L1-L4, minimax, mixed)
│   ├── wrappers.py         # FlattenAndNormalizeObs (/2.0 + mask forwarding)
│   ├── evaluation.py       # env factories + eval matrix vs opponent pool
│   ├── gui.py              # Pygame rendering component
│   └── utils.py            # board_to_obs, predict_action, load_model, ASCII render
├── scripts/              # CLI entrypoints (run with uv run python scripts/<name>.py)
│   ├── train.py            # MaskablePPO curriculum training
│   ├── eval.py             # standalone eval matrix
│   ├── play.py             # CLI play vs trained agent
│   └── play_gui.py         # Pygame GUI play vs trained agent
├── models/               # generated artifacts (overwritten by scripts/train.py)
│   ├── ppo_tictactoe.zip
│   └── ppo_eval.txt
```

## Requirements

- Python >= 3.12
- Poetry or uv (for dependency management)

### Dependencies

```
numpy>=2.4.4
pygame>=2.6.1
stable-baselines3>=2
sb3-contrib>=2
gymnasium
```

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Or install manually
uv pip install numpy pygame stable-baselines3 gymnasium
```

### Using pip

```bash
pip install numpy pygame stable-baselines3 gymnasium
```

## Training

Train a MaskablePPO agent (default 1M-step curriculum: random 30% → rule-L4 30% → mixed 40%):

```bash
uv run python scripts/train.py
# quick sanity run:
uv run python scripts/train.py --smoke
# single opponent / no curriculum:
uv run python scripts/train.py --opponent minimax --no-randomize-first
```

### Training Options (`uv run python scripts/train.py --help`)

```bash
--timesteps 1000000 --n-envs 8 --seed 42 --save-path models/ppo_tictactoe
--opponent {random,rule_l4,minimax,mixed}  # default: curriculum
--no-curriculum --no-randomize-first --eval-episodes 300
```

### Training Details

- **Algorithm**: MaskablePPO (sb3-contrib), `device="cpu"`
- **Policy**: MLP `net_arch=dict(pi=[128, 128], vf=[128, 128])`, `n_steps=1024`, `batch_size=512`, `ent_coef=0.01`
- **Masking**: `TicTacToeEnv.action_masks()` + `ActionMasker`; illegal-move rate should be 0
- **Sides**: `randomize_first=True` — agent trains as X-first and O-second
- **Curriculum**: `random` → `rule_l4` (win+block) → `mixed` (minimax-heavy); model carries over via `set_env`
- **Observation**: 3x3 board flattened and normalized to [0, 1] via `utils.board_to_obs`
  - Empty: 0.0
  - Agent (X): 0.5
  - Opponent (O): 1.0
- **Reward Structure**:
  - Win: +1.0
  - Loss: -1.0
  - Draw: +0.5
  - Illegal move: -1.0
  - Step penalty: -0.01
  - Attack bonus (create 2-in-a-row): +0.1
  - Defense bonus (block opponent win): +0.1

## Playing Against the Agent

### CLI Mode

```bash
python scripts/play.py
```

**Controls:**
- Enter move number (0-8 or 1-9) to play
- `y`/`n` to answer prompts
- Ctrl+C to quit

### GUI Mode

```bash
python scripts/play_gui.py
```

**Controls:**
- Click on empty squares to make a move
- ESC key or window close to quit
- Click anywhere to continue after game ends

## Opponent Policies

The `tic_tac_toe_rl/opponents.py` module provides various opponent strategies:

| Policy | Description | Difficulty |
|--------|-------------|------------|
| `random_policy` | Random legal moves | Easy |
| `rule_based_policy(level=1)` | Level 1: Random | Easy |
| `rule_based_policy(level=2)` | Prefers center and corners | Medium-Easy |
| `rule_based_policy(level=3)` | Blocks opponent wins | Medium |
| `rule_based_policy(level=4)` | Takes winning moves + blocks | Medium-Hard |
| `minimax_policy` | Optimal play (unbeatable) | Hard |
| `minimax_randomized_policy` | Minimax with random probability | Configurable |

### Custom Opponent Example

```python
from tic_tac_toe_rl.tic_tac_toe_env import TicTacToeEnv
from tic_tac_toe_rl.opponents import rule_based_policy, minimax_policy

# Create environment with rule-based opponent (level 3)
env = TicTacToeEnv(opponent_policy=lambda b: rule_based_policy(b, level=3))

# Or use minimax (unbeatable)
env = TicTacToeEnv(opponent_policy=minimax_policy)
```

## Reward Shaping

The environment includes reward shaping to guide learning:

- **Attack Reward (+0.1)**: Awarded when the agent creates a 2-in-a-row with an empty third position
- **Defense Reward (+0.1)**: Awarded when the agent blocks the opponent's immediate winning move

These shaping rewards help the agent learn good strategies faster without waiting for the final game outcome.

## Evaluation

After training, the model is evaluated on a matrix (agent-first / opponent-first / mixed starts):

```bash
uv run python scripts/eval.py --episodes 300
# Results saved to models/ppo_eval.txt, e.g.:
#   vs random     agent=... | opponent=... | random=...
#   vs rule_l4    agent=... | opponent=... | random=...
#   vs minimax    agent=... | opponent=... | random=...
```

Target for a strong agent: beat random and rule-L4 consistently, draw minimax
(perfect tic-tac-toe is a draw). Illegal-move count must be 0 (masking).

## Curriculum Learning

Built in: `scripts/train.py` runs random → rule-L4 → mixed by default. Single-opponent
runs via `--opponent <name> --no-curriculum`. Opponent pool lives in
`tic_tac_toe_rl/opponents.py: get_opponent_policy('random' | 'rule_l4' | 'minimax' | 'mixed')`.

## Self-Play Training (Advanced)

To enable self-play, the agent should train against itself:

```python
from utils import board_to_obs, predict_action, load_model

model = load_model("models/ppo_tictactoe.zip")

def self_play_policy(board):
    return predict_action(model, board, deterministic=False)

env = TicTacToeEnv(opponent_policy=self_play_policy, randomize_first=True)
```

## File Descriptions

| File | Purpose |
|------|---------|
| `tic_tac_toe_rl/tic_tac_toe_env.py` | Gymnasium environment with reward shaping, masking, both-sides starts |
| `tic_tac_toe_rl/opponents.py` | Opponent pool: random, rule-based L1-L4, minimax (cached), mixed |
| `tic_tac_toe_rl/wrappers.py` | `FlattenAndNormalizeObs` (obs `/2.0` + mask forwarding) |
| `tic_tac_toe_rl/evaluation.py` | Env factories + eval matrix shared by `train.py` / `eval.py` |
| `scripts/train.py` | MaskablePPO curriculum training + eval matrix |
| `scripts/eval.py` | Standalone eval matrix vs opponent pool |
| `scripts/play.py` | CLI interface to play against trained agent |
| `scripts/play_gui.py` | Pygame GUI interface to play against trained agent |
| `tic_tac_toe_rl/gui.py` | Pygame rendering component |
| `tic_tac_toe_rl/utils.py` | Shared helpers: `board_to_obs`, `predict_action`, `load_model`, ASCII render |

## Tips

1. **Training Time**: For best results, train with at least 500,000-1,000,000 timesteps
2. **Parallel Environments**: Use `n_envs=8` or higher for faster training
3. **Deterministic vs Stochastic**: The agent can play deterministically (always picks best move) or stochastically (samples from policy)
4. **Model Saving**: Models are saved with `.zip` extension and can be loaded later

## Troubleshooting

### Common Issues

- **ModuleNotFoundError**: Run `uv sync`
- **Model loading fails**: retrain with current `train.py` (legacy PPO checkpoints load via fallback, but retraining to MaskablePPO is recommended)
- **Pygame errors**: needs a display; `play.py` (CLI) works headless, `play_gui.py` exits with an error when headless

### Verification

Check all dependencies are installed:
```bash
uv pip list
# or
pip list
```

## License

MIT License - see LICENSE file for details.
