# Google Colab Training Setup Guide

This guide explains how to train your custom emergency vehicle yielding environment on Google Colab to leverage free GPU acceleration.

## 🚀 NEW: OPTIMIZED VERSION AVAILABLE!

**Use `colab_training_optimized.ipynb` for 6-10x faster training!**
- **Original**: 10-20 hours on GPU, 60 hours on CPU
- **Optimized**: 2-3 hours on GPU, 6-10 hours on CPU

The optimized version uses:
- ✅ 4 parallel environments (4x speedup)
- ✅ 25 vehicles instead of 50 (2x speedup)
- ✅ 30s episodes instead of 40s (1.3x speedup)
- ✅ Better hyperparameters for faster convergence

**Recommendation: Use `colab_training_optimized.ipynb` instead of the original!**

---

## Quick Start

### 1. Upload Files to Google Drive

Create a folder in Google Drive (e.g., `CS272_Project`) and upload these files:
- `custom_env/custom_env/emergency_env.py`
- `custom_env/custom_emergency_lidar.py` (or your training script)
- `custom_env/custom_emergency_grayscale.py` (if needed)

### 2. Create a Colab Notebook

Open Google Colab (https://colab.research.google.com) and create a new notebook.

### 3. Setup Code Cells

#### Cell 1: Mount Google Drive and Install Dependencies
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install gymnasium highway-env stable-baselines3[extra] pandas matplotlib tqdm

# Verify GPU is available
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

#### Cell 2: Setup Custom Environment
```python
import sys
import os

# Add custom environment to path
sys.path.insert(0, '/content/drive/MyDrive/CS272_Project')

# Create custom_env module structure
os.makedirs('/content/custom_env', exist_ok=True)

# Copy emergency_env.py
!cp /content/drive/MyDrive/CS272_Project/emergency_env.py /content/custom_env/

# Create __init__.py
with open('/content/custom_env/__init__.py', 'w') as f:
    f.write('')

# Verify import works
import custom_env.emergency_env
print("Custom environment imported successfully!")
```

#### Cell 3: Training Script
```python
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup directories
SAVE_DIR = "/content/drive/MyDrive/CS272_Project/models"
LOG_DIR = "/content/drive/MyDrive/CS272_Project/logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure environment with LiDAR observation
config = {
    "observation": {
        "type": "LidarObservation",
        "cells": 64,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
}

def make_env():
    env = gym.make("EmergencyHighwayEnv-v0", config=config, render_mode=None)
    env = Monitor(env, filename=f"{LOG_DIR}/monitor_emergency_lidar.csv")
    return env

# Create vectorized environment
venv = DummyVecEnv([make_env])

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=SAVE_DIR,
    name_prefix="ppo_emergency_lidar_checkpoint"
)

# Evaluation callback
eval_env = DummyVecEnv([make_env])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=25_000,
    deterministic=True,
    render=False
)

# Create PPO model - use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

model = PPO(
    "MlpPolicy",
    venv,
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    device=device,
    tensorboard_log=f"{LOG_DIR}/tb/"
)

# Train the model
print("Starting training for Emergency Vehicle Yielding Environment (LiDAR)...")
print("Training for ~4000 episodes (500,000 timesteps)...")
model.learn(
    total_timesteps=500_000,
    tb_log_name="run_emergency_lidar",
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# Save final model
final_path = f"{SAVE_DIR}/ppo_emergency_lidar_final"
model.save(final_path)
print(f"Training done. Model saved to: {final_path}")
```

#### Cell 4: Plot Learning Curve
```python
def plot_learning_curve(log_path, output_path):
    df = pd.read_csv(log_path, skiprows=1)
    rewards = df["r"].values
    window = 20
    smoothed = pd.Series(rewards).rolling(window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw episodic reward", color='blue')
    plt.plot(smoothed, linewidth=2, label=f"Smoothed (window={window})", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve - Emergency Yielding (LiDAR Observation)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Learning curve saved to: {output_path}")
    plt.show()

learning_curve_path = f"{LOG_DIR}/emergency_lidar_learning_curve.png"
plot_learning_curve(f"{LOG_DIR}/monitor_emergency_lidar.csv", learning_curve_path)
```

#### Cell 5: Evaluate Model
```python
# Load best model
print("Loading best model for evaluation...")
model = PPO.load(f"{SAVE_DIR}/best_model")

def evaluate_agent(model, make_env_fn, episodes=500):
    returns = []
    env = make_env_fn()

    for ep in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        returns.append(total_reward)

        if (ep + 1) % 100 == 0:
            print(f"Evaluated {ep + 1}/{episodes} episodes...")

    env.close()
    return returns

print("Running 500-episode deterministic evaluation...")
returns = evaluate_agent(model, make_env)

# Violin plot for performance test
plt.figure(figsize=(7, 6))
parts = plt.violinplot([returns], showmeans=True, showextrema=True)
plt.xticks([1], ["PPO (LiDAR)"])
plt.ylabel("Episodic Return")
plt.title("Performance Test - Emergency Yielding (LiDAR, 500 episodes)")
plt.grid(axis="y")
plt.tight_layout()

performance_path = f"{LOG_DIR}/emergency_lidar_performance_test.png"
plt.savefig(performance_path, dpi=300)
print(f"Performance test plot saved to: {performance_path}")
plt.show()

print(f"\n=== Final Results ===")
print(f"Mean return over 500 episodes: {np.mean(returns):.2f}")
print(f"Std return: {np.std(returns):.2f}")
print(f"Min return: {np.min(returns):.2f}")
print(f"Max return: {np.max(returns):.2f}")
```

## Important Notes

### GPU Runtime
1. In Colab, go to `Runtime` → `Change runtime type`
2. Select `T4 GPU` or `GPU` for Hardware accelerator
3. Click `Save`

### Session Management
- Colab free tier has session limits (~12 hours)
- Models and logs save to Google Drive automatically
- You can resume training by loading the latest checkpoint:
  ```python
  # Load the latest checkpoint
  model = PPO.load(f"{SAVE_DIR}/ppo_emergency_lidar_checkpoint_XXXXX_steps")
  # Continue training
  model.learn(total_timesteps=500_000, reset_num_timesteps=False)
  ```

### Monitoring Training
- Use TensorBoard in Colab:
  ```python
  %load_ext tensorboard
  %tensorboard --logdir {LOG_DIR}/tb/
  ```

### Download Results
All models, logs, and plots are automatically saved to Google Drive in the `CS272_Project` folder.

## Advantages of Training on Colab

1. **GPU Acceleration**: 5-10x faster training compared to CPU
2. **Free Resources**: No cost for reasonable usage
3. **No Local Setup**: Works directly in browser
4. **Persistent Storage**: Models saved to Google Drive
5. **Easy Sharing**: Share notebook with collaborators

## Troubleshooting

### Out of Memory
If you get OOM errors, reduce batch_size:
```python
model = PPO(
    "MlpPolicy",
    venv,
    batch_size=128,  # Reduced from 256
    # ... other params
)
```

### Session Timeout
Set up checkpoints more frequently:
```python
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,  # Save every 10k steps instead of 20k
    save_path=SAVE_DIR,
    name_prefix="ppo_emergency_lidar_checkpoint"
)
```

### Import Errors
Make sure the custom environment path is correctly set and the file is uploaded to Google Drive.
