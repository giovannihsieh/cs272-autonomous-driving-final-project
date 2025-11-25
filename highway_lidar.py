import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import DQN, PPO
import gymnasium as gym
import highway_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# Make environment function (for evaluation)
def make_env():
    env = gym.make(
        "highway-v0",
        render_mode=None,
        config={"observation": {"type": "LidarObservation", "cells": 128}}
    )
    return Monitor(env)

# define models (DQN and PPO)
model_DQN = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=64,
    tau=0.1,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./logs/DQN/"
)

model_PPO = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/PPO/"
)


# train models (DQN and PPO)
model_DQN.learn(total_timesteps=200_000, tb_log_name="run_DQN")
model_DQN.save("highway_lidar_dqn")

model_PPO.learn(total_timesteps=200_000, tb_log_name="run_PPO")
model_PPO.save("highway_lidar_ppo")

# Load models
model_DQN = DQN.load("highway_lidar_dqn")
model_PPO = PPO.load("highway_lidar_ppo")

# Plot learning curve from CSV (already generated)
def plot_learning_curve(log_file, label):
    df = pd.read_csv(log_file)  # <- no skiprows
    rewards = df["r"].rolling(20).mean()  # smoothed
    episodes = np.arange(len(df))
    plt.plot(episodes, rewards, label=label)

plt.figure(figsize=(10, 6))
plot_learning_curve("./logs/DQN/monitor.csv", "DQN")
plot_learning_curve("./logs/PPO/monitor.csv", "PPO")
plt.xlabel("Episodes")
plt.ylabel("Mean Episodic Reward (smoothed)")
plt.title("Learning Curve — highway-v0 (Lidar)")
plt.legend()
plt.show()


# Evaluate models for violin plot
def evaluate(model, env_fn, episodes=500):
    returns = []
    env = env_fn()
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
    return returns

dqn_returns = evaluate(model_DQN, make_env, episodes=500)
ppo_returns = evaluate(model_PPO, make_env, episodes=500)

# Violin plot of evaluation performance
plt.figure(figsize=(8, 6))
plt.violinplot([dqn_returns, ppo_returns], showmeans=True, showextrema=True)
plt.xticks([1, 2], ["DQN", "PPO"])
plt.ylabel("Episodic Return")
plt.title("Evaluation Performance (500 episodes, deterministic policy)")
plt.show()
