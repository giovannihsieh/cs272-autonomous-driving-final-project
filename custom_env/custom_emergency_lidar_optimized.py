"""
Optimized Training script for Emergency Vehicle Yielding Environment
Optimizations:
1. Parallel environments using SubprocVecEnv (4-8x speedup)
2. Reduced vehicle count (50 -> 25, ~2x speedup)
3. GPU acceleration
4. Shorter episodes (40s -> 30s, ~1.3x speedup)
5. Optimized hyperparameters
6. Less frequent evaluation

Expected speedup: 60 hours -> 6-10 hours
"""

import gymnasium as gym
import sys
import os
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import custom_env.emergency_env
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./custom_emergency_models_lidar_optimized"
LOG_DIR = "./custom_emergency_logs_lidar_optimized"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Optimized environment configuration
config = {
    "observation": {
        "type": "LidarObservation",
        "cells": 64,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "vehicles_count": 25,  # Reduced from 50 (2x speedup)
    "duration": 30,  # Reduced from 40 (1.3x speedup)
    "vehicles_density": 1.0,  # Keep same density
}

def make_env(rank):
    """Create a single environment (for parallel workers)"""
    def _init():
        env = gym.make("EmergencyHighwayEnv-v0", config=config, render_mode=None)
        env = Monitor(env, filename=None)  # We'll monitor the vectorized env instead
        return env
    return _init

# Detect number of CPUs for parallel environments
num_cpu = os.cpu_count()
num_envs = min(8, num_cpu - 2) if num_cpu else 4  # Use 4-8 parallel envs
print(f"Using {num_envs} parallel environments (detected {num_cpu} CPUs)")

# Create parallel vectorized environments (4-8x speedup!)
print("Creating parallel environments...")
venv = SubprocVecEnv([make_env(i) for i in range(num_envs)])
venv = VecMonitor(venv, filename=f"{LOG_DIR}/monitor_emergency_lidar_optimized.csv")

# Checkpoint callback - save less frequently (40k vs 20k)
checkpoint_callback = CheckpointCallback(
    save_freq=40_000 // num_envs,  # Adjust for parallel envs
    save_path=SAVE_DIR,
    name_prefix="ppo_emergency_lidar_opt_checkpoint"
)

# Evaluation callback - evaluate less frequently (50k vs 25k)
eval_env = SubprocVecEnv([make_env(i) for i in range(2)])  # Use 2 envs for eval
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=50_000 // num_envs,  # Adjust for parallel envs
    deterministic=True,
    render=False,
    n_eval_episodes=10  # Reduced from default 5 for better estimates
)

# Detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training device: {device}")
if device == "cpu":
    print("WARNING: No GPU detected. Training will be slower. Consider using Google Colab with GPU.")

# Create PPO model with optimized hyperparameters
# Adjusted for faster convergence and parallel environments
model = PPO(
    "MlpPolicy",
    venv,
    learning_rate=3e-4,  # Slightly higher for faster learning
    n_steps=2048 // num_envs,  # Adjust for parallel envs (collect same total steps)
    batch_size=256,
    n_epochs=10,  # Increased from 5 for better sample efficiency
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,  # Standard value
    ent_coef=0.01,  # Increased for more exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    device=device,
    tensorboard_log=f"{LOG_DIR}/tb/"
)

# Train the model
print("\n" + "="*60)
print("OPTIMIZED TRAINING FOR EMERGENCY VEHICLE ENVIRONMENT")
print("="*60)
print(f"Parallel environments: {num_envs}")
print(f"Vehicles per env: 25 (reduced from 50)")
print(f"Episode duration: 30s (reduced from 40s)")
print(f"Total timesteps: 500,000")
print(f"Device: {device}")
print(f"Expected time: ~6-10 hours (vs 60 hours original)")
print("="*60 + "\n")

model.learn(
    total_timesteps=500_000,
    tb_log_name="run_emergency_lidar_optimized",
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# Save final model
final_path = f"{SAVE_DIR}/ppo_emergency_lidar_optimized_final"
model.save(final_path)
print(f"\nTraining complete! Model saved to: {final_path}")

# Clean up parallel envs
venv.close()
eval_env.close()

# Plot learning curve
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
    plt.title("Learning Curve - Emergency Yielding (LiDAR, Optimized)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Learning curve saved to: {output_path}")
    plt.close()

learning_curve_path = f"{LOG_DIR}/emergency_lidar_optimized_learning_curve.png"
plot_learning_curve(f"{LOG_DIR}/monitor_emergency_lidar_optimized.csv", learning_curve_path)

# Load best model for evaluation
print("\nLoading best model for evaluation...")
model = PPO.load(f"{SAVE_DIR}/best_model")

# Evaluate over 500 episodes
def evaluate_agent(model, config, episodes=500):
    returns = []
    env = gym.make("EmergencyHighwayEnv-v0", config=config, render_mode=None)

    print(f"Running {episodes}-episode evaluation...")
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
            print(f"  Evaluated {ep + 1}/{episodes} episodes...")

    env.close()
    return returns

print("\nRunning 500-episode deterministic evaluation...")
returns = evaluate_agent(model, config, episodes=500)

# Violin plot for performance test
plt.figure(figsize=(7, 6))
parts = plt.violinplot([returns], showmeans=True, showextrema=True)
plt.xticks([1], ["PPO (LiDAR, Optimized)"])
plt.ylabel("Episodic Return")
plt.title("Performance Test - Emergency Yielding (Optimized, 500 episodes)")
plt.grid(axis="y")
plt.tight_layout()

performance_path = f"{LOG_DIR}/emergency_lidar_optimized_performance_test.png"
plt.savefig(performance_path, dpi=300)
print(f"Performance test plot saved to: {performance_path}")
plt.close()

print(f"\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Mean return over 500 episodes: {np.mean(returns):.2f}")
print(f"Std return: {np.std(returns):.2f}")
print(f"Min return: {np.min(returns):.2f}")
print(f"Max return: {np.max(returns):.2f}")
print(f"\nPlots saved:")
print(f"  - {learning_curve_path}")
print(f"  - {performance_path}")
print(f"  - Model: {final_path}.zip")
print("="*60)
