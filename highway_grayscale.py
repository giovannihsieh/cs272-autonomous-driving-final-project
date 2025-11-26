import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "policy_frequency": 2,
}

def make_env():
    os.makedirs("logs", exist_ok=True)
    env = gym.make("highway-v0", config=config, render_mode = None)
    env = Monitor(env, filename="logs/monitor.csv")  # write file explicitly
    return env

# PPO needs a VecEnv
venv = DummyVecEnv([make_env])

# Create PPO model with cnnpolicy for image observations
model = PPO(
    "CnnPolicy",
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
    tensorboard_log="./logs/highway_grayscale/",
    device="cuda" # use GPU
)

# Train the model
print("Starting training...")
model.learn(total_timesteps=200000, tb_log_name="run_highway_grayscale")
model.save("ppo_highway_grayscale")

print("Training done, model saved as 'highway_grayscale'")

def plot_learning_curve(log_path, label="PPO"):
    # monitor.csv created by Monitor wrapper
    df = pd.read_csv(log_path, skiprows=1)   # Skip header comment line
    rewards = df["r"].values

    # Smooth rewards (moving average)
    window = 20
    smoothed = pd.Series(rewards).rolling(window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw episodic reward")
    plt.plot(smoothed, linewidth=2, label=f"Smoothed (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve — PPO (GrayscaleObservation)")
    plt.legend()
    plt.grid()
    plt.show()


# Plot learning curve
plot_learning_curve("./logs/monitor.csv")

# evaluate over 500 episodes
def evaluate_agent(model, make_env_fn, episodes=200):
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
plt.xticks([1], ["PPO"])
plt.ylabel("Episodic Return")
plt.title("Performance of PPO (500 deterministic episodes)")
plt.grid(axis="y")
plt.show()

print(f"Mean return over 500 episodes: {np.mean(returns):.2f}")
print(f"Std return: {np.std(returns):.2f}")

