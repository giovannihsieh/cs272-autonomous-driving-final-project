"""
Training script for Emergency Vehicle Yielding Environment with Grayscale Observation
ID 13: Learning curve for custom environment (Grayscale variant)
ID 14: Performance test for custom environment (Grayscale variant)

Based on PPO implementation from stable-baselines3
https://github.com/DLR-RM/stable-baselines3
"""

import gymnasium as gym
import sys
import os

# Add current directory to path to import custom_env module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import custom_env.emergency_env
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = "./custom_emergency_models_grayscale"
LOG_DIR = "./custom_emergency_logs_grayscale"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure environment with Grayscale observation
config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # RGB to grayscale weights
        "scaling": 1.75,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
}

def make_env():
    env = gym.make("EmergencyHighwayEnv-v0", config=config, render_mode=None)
    env = Monitor(env, filename=f"{LOG_DIR}/monitor_emergency_grayscale.csv")
    return env

# Create vectorized environment
venv = DummyVecEnv([make_env])

# Checkpoint callback to save model periodically
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=SAVE_DIR,
    name_prefix="ppo_emergency_grayscale_checkpoint"
)

# Evaluation callback to save best model
eval_env = DummyVecEnv([make_env])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=25_000,
    deterministic=True,
    render=False
)

# Create PPO model with CnnPolicy for Grayscale images
model = PPO(
    "CnnPolicy",
    venv,
    learning_rate=1e-4,  # Lower learning rate for CNN
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
    tensorboard_log=f"{LOG_DIR}/tb/"
)

# Train the model
print("Starting training for Emergency Vehicle Yielding Environment (Grayscale)...")
print("Training for ~4000 episodes (500,000 timesteps)...")
model.learn(
    total_timesteps=500_000,
    tb_log_name="run_emergency_grayscale",
    callback=[checkpoint_callback, eval_callback]
)

# Save final model
final_path = f"{SAVE_DIR}/ppo_emergency_grayscale_final"
model.save(final_path)
print(f"Training done. Model saved to: {final_path}")

# Plot learning curve (ID 13)
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
    plt.title("ID 13: Learning Curve - Emergency Yielding (Grayscale Observation)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Learning curve saved to: {output_path}")
    plt.close()

learning_curve_path = f"{LOG_DIR}/ID_13_emergency_grayscale_learning_curve.png"
plot_learning_curve(f"{LOG_DIR}/monitor_emergency_grayscale.csv", learning_curve_path)

# Load best model for evaluation
print("\nLoading best model for evaluation...")
model = PPO.load(f"{SAVE_DIR}/best_model")

# Evaluate over 500 episodes (ID 14)
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

print("\nRunning 500-episode deterministic evaluation...")
returns = evaluate_agent(model, make_env)

# Violin plot for performance test (ID 14)
plt.figure(figsize=(7, 6))
parts = plt.violinplot([returns], showmeans=True, showextrema=True)
plt.xticks([1], ["PPO (Grayscale)"])
plt.ylabel("Episodic Return")
plt.title("ID 14: Performance Test - Emergency Yielding (Grayscale, 500 episodes)")
plt.grid(axis="y")
plt.tight_layout()

performance_path = f"{LOG_DIR}/ID_14_emergency_grayscale_performance_test.png"
plt.savefig(performance_path, dpi=300)
print(f"Performance test plot saved to: {performance_path}")
plt.close()

print(f"\n=== Final Results ===")
print(f"Mean return over 500 episodes: {np.mean(returns):.2f}")
print(f"Std return: {np.std(returns):.2f}")
print(f"Min return: {np.min(returns):.2f}")
print(f"Max return: {np.max(returns):.2f}")
print(f"\nPlots saved:")
print(f"  - {learning_curve_path}")
print(f"  - {performance_path}")
