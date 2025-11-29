import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

SAVE_DIR = "./highway_models"
LOG_DIR = "./highway_logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

import os
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

config = {
    "observation": {
        "type": "LidarObservation",
    }
}

def make_env():
    env = gym.make("highway-v0", config=config, render_mode=None)
    # don't override the grayscale monitor file
    env = Monitor(env, filename=f"{LOG_DIR}/monitor_lidar.csv")
    return env

# PPO needs a VecEnv
venv = DummyVecEnv([make_env])

# checkpoints to save model
checkpoint_callback = CheckpointCallback(
    save_freq=20_000, # save every 20k steps
    save_path=SAVE_DIR,
    name_prefix="ppo_checkpoint"
)

# keep the best model
eval_env = DummyVecEnv([make_env])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=25_000, # evaluate best model every 25k steps
    deterministic=True,
    render=False
)

# Create PPO model with mlppolicy
model = PPO(
    "MlpPolicy",
    venv,
    learning_rate=2e-4, # went from 3e to 1e for smoother training
    n_steps=2048, # want to run 4096 if possible
    batch_size=256, # increased from 64 to 256 for CNN stability
    n_epochs=5, # reduced from 10 to 5 bc training was a bit unstable
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1, # from 0.2 -> 0.1 for smaller clip for image input
    ent_coef=0.001, # lower to avoid over-exploration bc it's CNN
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=f"{LOG_DIR}/tb/"
)

# Train the model
print("Starting training...")
model.learn(
    total_timesteps=200_000,
    tb_log_name="run_highway_lidar",
    callback=[checkpoint_callback, eval_callback]
)

# save final model
final_path = f"{SAVE_DIR}/ppo_highway_lidar_final"
model.save(final_path)
print(f"Training done. Model saved to: {final_path}")

def plot_learning_curve(log_path, label="PPO-LiDAR"):
    df = pd.read_csv(log_path, skiprows=1)
    rewards = df["r"].values
    window = 20
    smoothed = pd.Series(rewards).rolling(window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw episodic reward")
    plt.plot(smoothed, linewidth=2, label=f"Smoothed (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve — PPO (LiDARObservation)")
    plt.legend()
    plt.grid()
    plt.show()

plot_learning_curve(f"{LOG_DIR}/monitor_lidar.csv", label="PPO-LiDAR")

from stable_baselines3 import PPO
# final model
# model = PPO.load(f"{SAVE_DIR}/ppo_highway_lidar_final")
# best model
model = PPO.load(f"{SAVE_DIR}/best_model")

# evaluate over 500 episodes
def evaluate_agent(model, make_env_fn, episodes=500):
    returns = []
    env = make_env_fn()

    for _ in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        returns.append(total_reward)

    env.close()
    return returns

print("\nRunning 500-episode deterministic evaluation...")
returns = evaluate_agent(model, make_env)

# violin plot
plt.figure(figsize=(7, 6))
plt.violinplot([returns], showmeans=True, showextrema=True)
plt.xticks([1], ["PPO (LiDAR)"])
plt.ylabel("Episodic Return")
plt.title("Performance of PPO (500 deterministic episodes)")
plt.grid(axis="y")
plt.show()

print(f"Mean return over 500 episodes: {np.mean(returns):.2f}")
print(f"Std return: {np.std(returns):.2f}")