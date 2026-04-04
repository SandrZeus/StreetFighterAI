# Reinforcement Learning for Player-Specific Strategy Adaptation in Fighting Games

Bachelor's thesis project — training a PPO-based reinforcement learning agent to play Street Fighter II using frame-level visual input.

## Overview

The agent learns to play Street Fighter II: Special Champion Edition (Genesis) entirely from pixel observations. Raw game frames are preprocessed into 84x84 grayscale images, and the agent trains on frame deltas (difference between consecutive frames) to detect motion and action patterns.

The training pipeline uses Optuna for automated hyperparameter optimization across 10 trials, followed by extended training with the best parameters for 10M+ total timesteps.

## Results

Two training runs were conducted and compared:

**3M timesteps** — The agent adopted a predominantly defensive strategy (crouching), occasionally landing low kicks. It won some rounds but couldn't consistently win full matches. The reward curve showed gradual improvement, indicating the agent was slowly gaining confidence.

**7M timesteps** — The agent learned to block attacks, execute multi-hit leg combos, and occasionally stand up to throw punches. It defeated the game's first built-in AI opponent and advanced to the second. Performance dropped against the new opponent (different strategy), but an upward trend resumed after ~6M steps, suggesting it would continue improving with more training time.

Key observations:
- The agent developed emergent behaviors (blocking, combo chaining, retreating under pressure) without any hard-coded strategy
- Action diversity was limited — the agent converged on leg-based attacks from a crouching position, likely because early rewards reinforced this as the safest approach
- Performance was sensitive to opponent changes, highlighting the challenge of generalization in RL

## Limitations

This thesis is honest about what it doesn't achieve. The title mentions "player-specific adaptation," but the current implementation trains against scripted AI opponents only — no real human gameplay data was used. There is no opponent modeling, no online learning, and no continual adaptation during gameplay. The thesis frames this as groundwork and future direction, not a solved problem. Hardware constraints (laptop GPU) also limited training scale.

## How It Works

1. **Custom Gym environment** wraps the Retro emulator with preprocessed observations (grayscale, resize, frame delta) and a reward function based on score delta
2. **Hyperparameter search** (`StreetFighter.py`) runs 10 Optuna trials, each training for 3M timesteps and evaluating over 5 episodes
3. **Training** (`Train.py`) loads the best trial's parameters and model, then continues training for 3M additional timesteps with periodic checkpointing
4. **Extended training** (`7mil.py`) continues from the trained model for another 7M timesteps
5. **Evaluation** (`RenderTest.py`) loads a checkpoint and renders 5 episodes with mean reward reporting

## Tech Stack

- Python, OpenAI Gym, Retro Gym (Genesis emulator)
- Stable Baselines3 (PPO with CnnPolicy)
- Optuna (hyperparameter optimization)
- PyTorch (CUDA-accelerated training)
- OpenCV (frame preprocessing)

## Best Hyperparameters (Optuna Trial 7)

| Parameter | Value |
|---|---|
| n_steps | 7989 |
| gamma | 0.9126 |
| learning_rate | 1.27e-05 |
| clip_range | 0.2635 |
| gae_lambda | 0.9096 |

## Project Structure

```
├── StreetFighter.py      # Optuna hyperparameter search (10 trials × 3M steps)
├── Train.py              # Training with best params (3M steps)
├── 7mil.py               # Continued training (7M steps)
├── RenderTest.py         # Visual evaluation of trained agent
├── StreetFighter.ipynb   # Jupyter notebook version of the full pipeline
├── Bachelor_s_Thesis.pdf # Full thesis document
├── opt/                  # Saved models from Optuna trials
├── trained/              # Final trained model checkpoints
└── roms/                 # Game ROM (not included)
```

## Setup

```bash
pip install gym==0.21.0 gym-retro==0.7.1 stable-baselines3==1.6.2 optuna==2.10.1 opencv-python torch
```

Place the Street Fighter II ROM in the `roms/` directory and import it:

```bash
cd roms
python -m retro.import .
```

## Run

```bash
# Hyperparameter search
python StreetFighter.py

# Train with best parameters
python Train.py

# Continue training
python 7mil.py

# Evaluate with rendering
python RenderTest.py
```

## Thesis

The full thesis document is included in this repository: [Bachelor's Thesis (PDF)](Bachelor_s_Thesis.pdf)

## Note

The `.py` and `.ipynb` versions of the main pipeline contain the same code. The `.py` version exists because Jupyter rendering had compatibility issues with Hyprland (Wayland compositor).
