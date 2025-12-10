"""
Highway Environment - DRL Training
Task 1: IDs 1-4 (Highway with LiDAR and Grayscale observations)

Usage:
    python highway_training.py --obs lidar --timesteps 100000
    python highway_training.py --obs grayscale --timesteps 150000
    python highway_training.py --obs both --timesteps 100000
"""

import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import os
import argparse

# =============================================================================
# CONFIGURATIONS
# =============================================================================

LOG_DIR = "./logs_highway"

# LiDAR Config (IDs 1, 2)
lidar_config = {
    "observation": {
        "type": "LidarObservation",
        "cells": 64,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 15,
    "duration": 40,
    "simulation_frequency": 10,
    "policy_frequency": 2,
}

# Grayscale Config (IDs 3, 4)
grayscale_config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (64, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 15,
    "duration": 40,
    "simulation_frequency": 10,
    "policy_frequency": 2,
}

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_highway(obs_type="lidar", total_timesteps=100000):
    """Train PPO on Highway environment."""
    
    print(f"\n{'='*50}")
    print(f"🚗 Training Highway: {obs_type.upper()}")
    print(f"{'='*50}")
    
    if obs_type == "lidar":
        config = lidar_config
        policy = "MlpPolicy"
        save_dir = f"{LOG_DIR}/lidar"
    else:
        config = grayscale_config
        policy = "CnnPolicy"
        save_dir = f"{LOG_DIR}/grayscale"
    
    os.makedirs(save_dir, exist_ok=True)
    
    def make_env():
        env = gym.make("highway-v0", config=config, render_mode=None)
        env = Monitor(env, filename=f"{save_dir}/monitor.csv")
        return env
    
    vec_env = DummyVecEnv([make_env])
    
    model = PPO(
        policy,
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    model_path = f"{save_dir}/ppo_{obs_type}"
    model.save(model_path)
    print(f"✅ Saved: {model_path}")
    
    vec_env.close()
    return model, model_path

# =============================================================================
# EVALUATION FUNCTION (500 episodes)
# =============================================================================

def evaluate_500(model_path, obs_type="lidar"):
    """Evaluate trained model for 500 episodes without exploration."""
    
    print(f"\n📊 Evaluating {obs_type.upper()} for 500 episodes...")
    
    config = lidar_config if obs_type == "lidar" else grayscale_config
    env = gym.make("highway-v0", config=config, render_mode=None)
    model = PPO.load(model_path)
    
    rewards = []
    for ep in range(500):
        obs, _ = env.reset()
        done, truncated, total = False, False, 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total += reward
        rewards.append(total)
        if (ep + 1) % 100 == 0:
            print(f"  {ep+1}/500 | Mean: {np.mean(rewards):.2f}")
    
    env.close()
    print(f"\n✅ Final: Mean={np.mean(rewards):.2f}, Std={np.std(rewards):.2f}")
    return rewards

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_learning_curve(monitor_path, title, save_path, window=20):
    """Plot learning curve with raw and smoothed rewards."""
    
    df = pd.read_csv(monitor_path, skiprows=1)
    rewards = df['r'].values
    episodes = np.arange(len(rewards))
    rolling = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.4, color='#8884d8', linewidth=0.8, label='Raw episodic reward')
    plt.plot(episodes, rolling, color='orange', linewidth=2, label=f'Smoothed (window={window})')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_violin(rewards, title, save_path, xlabel="Highway"):
    """Plot violin plot for 500 episode evaluation."""
    
    plt.figure(figsize=(8, 6))
    parts = plt.violinplot([rewards], positions=[1], showmeans=True, showmedians=True, widths=0.8)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#87CEEB')
        pc.set_edgecolor('#4682B4')
        pc.set_alpha(0.7)
    for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        parts[key].set_color('#4682B4')
    
    plt.ylabel('Episodic Return', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks([1], [xlabel])
    
    mean_val = np.mean(rewards)
    std_val = np.std(rewards)
    margin = max(std_val * 3, 0.5)
    plt.ylim(mean_val - margin, mean_val + margin)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {save_path}")

# =============================================================================
# MAIN
# =============================================================================

def run_lidar(timesteps):
    """Run LiDAR training pipeline (IDs 1, 2)."""
    
    print("\n" + "="*60)
    print("🚀 LIDAR PIPELINE (IDs 1, 2)")
    print("="*60)
    
    # Train
    model, path = train_highway("lidar", timesteps)
    
    # Plot learning curve (ID 1)
    plot_learning_curve(
        f"{LOG_DIR}/lidar/monitor.csv",
        "Learning Curve - Highway (LidarObservation)",
        f"{LOG_DIR}/ID1_highway_lidar_learning_curve.png"
    )
    
    # Evaluate 500 episodes
    rewards = evaluate_500(path, "lidar")
    
    # Plot violin (ID 2)
    plot_violin(
        rewards,
        "Performance Test - Highway LidarObs (500 episodes)",
        f"{LOG_DIR}/ID2_highway_lidar_violin.png",
        "PPO (LidarObservation)"
    )
    
    return rewards


def run_grayscale(timesteps):
    """Run Grayscale training pipeline (IDs 3, 4)."""
    
    print("\n" + "="*60)
    print("🚀 GRAYSCALE PIPELINE (IDs 3, 4)")
    print("="*60)
    
    # Train
    model, path = train_highway("grayscale", timesteps)
    
    # Plot learning curve (ID 3)
    plot_learning_curve(
        f"{LOG_DIR}/grayscale/monitor.csv",
        "Learning Curve - Highway (GrayscaleObservation)",
        f"{LOG_DIR}/ID3_highway_grayscale_learning_curve.png"
    )
    
    # Evaluate 500 episodes
    rewards = evaluate_500(path, "grayscale")
    
    # Plot violin (ID 4)
    plot_violin(
        rewards,
        "Performance Test - Highway GrayscaleObs (500 episodes)",
        f"{LOG_DIR}/ID4_highway_grayscale_violin.png",
        "PPO (GrayscaleObservation)"
    )
    
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Highway environment")
    parser.add_argument("--obs", type=str, default="both", choices=["lidar", "grayscale", "both"],
                        help="Observation type to train")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Training timesteps")
    args = parser.parse_args()
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    if args.obs == "lidar":
        run_lidar(args.timesteps)
    elif args.obs == "grayscale":
        run_grayscale(args.timesteps)
    else:  # both
        run_lidar(args.timesteps)
        run_grayscale(int(args.timesteps * 1.5))  # Grayscale needs more steps
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutput files in {LOG_DIR}/:")
    print("  ID1_highway_lidar_learning_curve.png")
    print("  ID2_highway_lidar_violin.png")
    print("  ID3_highway_grayscale_learning_curve.png")
    print("  ID4_highway_grayscale_violin.png")


if __name__ == "__main__":
    main()
