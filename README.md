# Tic-Tac-Toe Reinforcement Learning

A reinforcement learning project for training a Tic-Tac-Toe playing agent using PPO (Proximal Policy Optimization) from Stable Baselines3.

## Features

- **Gymnasium Environment**: Custom `TicTacToeEnv` with reward shaping (attack/defense bonuses)
- **PPO Training**: Train agents using Stable Baselines3
- **Multiple Interfaces**: Play against trained agents via CLI or Pygame GUI
- **Opponent Policies**: Various opponent difficulty levels for curriculum learning
- **Normalized Observations**: Inputs scaled to [0, 1] for better training convergence

## Project Structure

```
tic-tac-toe-rl/
├── tic_tac_toe_env.py    # Gymnasium environment (core)
├── train.py              # PPO training script
├── play.py               # CLI play against trained agent
├── play_gui.py           # Pygame GUI play against trained agent
├── gui.py                # Pygame GUI component
├── opponents.py          # Various opponent policies (random, rule-based, minimax)
├── utils.py              # Shared utilities
├── ppo_tictactoe.zip     # Trained model (generated)
└── ppo_eval.txt          # Evaluation results (generated)
```

## Requirements

- Python >= 3.12
- Poetry or uv (for dependency management)

### Dependencies

```
numpy>=2.4.4
pygame>=2.6.1
stable-baselines3>=2
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

Train a PPO agent:

```bash
python train.py
```

### Training Options

```python
# In train.py, modify train_ppo() parameters:
train_ppo(
    total_timesteps=1_000_000,  # Total training steps
    n_envs=8,                    # Number of parallel environments
    save_path="ppo_tictactoe",   # Model save path
    seed=42,                     # Random seed
)
```

### Training Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MLP with architecture `[64, 64]` (input: 9 normalized board values, output: 9 action probabilities)
- **Opponent**: Random policy by default
- **Observation**: 3x3 board flattened and normalized to [0, 1]
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
python play.py
```

**Controls:**
- Enter move number (0-8 or 1-9) to play
- `y`/`n` to answer prompts
- Ctrl+C to quit

### GUI Mode

```bash
python play_gui.py
```

**Controls:**
- Click on empty squares to make a move
- ESC key or window close to quit
- Click anywhere to continue after game ends

## Opponent Policies

The `opponents.py` module provides various opponent strategies:

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
from tic_tac_toe_env import TicTacToeEnv
from opponents import rule_based_policy, minimax_policy

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

After training, the model is evaluated against 1000 games:

```bash
# Results are saved to ppo_eval.txt
# Format: Eval over 1000 games -> W/D/L: {wins}/{draws}/{losses}
```

Example output:
```
Eval over 1000 games -> W/D/L: 988/12/0
```

## Curriculum Learning (Advanced)

For better training, you can implement curriculum learning by gradually increasing opponent difficulty:

```python
from opponents import get_opponent_policy

# Create environments with different opponents
opponents = [
    get_opponent_policy('random'),
    get_opponent_policy('rulebased', level=2),
    get_opponent_policy('rulebased', level=3),
    get_opponent_policy('rulebased', level=4),
    get_opponent_policy('minimax'),
]
```

## Self-Play Training (Advanced)

To enable self-play, the agent should train against itself:

```python
# Load existing model and use it as opponent
model = PPO.load("ppo_tictactoe", device="cpu")

def self_play_policy(board):
    obs_vec = board.ravel().astype(np.float32) / 2.0  # Normalize
    action, _ = model.predict(obs_vec, deterministic=False)
    return int(action)

env = TicTacToeEnv(opponent_policy=self_play_policy)
```

## File Descriptions

| File | Purpose |
|------|---------|
| `tic_tac_toe_env.py` | Gymnasium environment with reward shaping |
| `train.py` | Training script with PPO configuration |
| `play.py` | CLI interface to play against trained agent |
| `play_gui.py` | Pygame GUI interface to play against trained agent |
| `gui.py` | Pygame rendering component |
| `opponents.py` | Various opponent policies for training/playing |
| `utils.py` | Shared utility functions |

## Tips

1. **Training Time**: For best results, train with at least 500,000-1,000,000 timesteps
2. **Parallel Environments**: Use `n_envs=8` or higher for faster training
3. **Deterministic vs Stochastic**: The agent can play deterministically (always picks best move) or stochastically (samples from policy)
4. **Model Saving**: Models are saved with `.zip` extension and can be loaded later

## Troubleshooting

### Common Issues

- **ModuleNotFoundError**: Run `uv sync` or `pip install -r pyproject.toml`
- **Model loading fails**: Ensure the model was saved with the same version of stable-baselines3
- **Pygame errors**: Make sure you have a display (use `-nw` flag for headless if needed)

### Verification

Check all dependencies are installed:
```bash
uv pip list
# or
pip list
```

## License

MIT License - see LICENSE file for details.
