import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from custom_env import ParallelParkingEnv

VIDEO_DIR = "./videos"
MODEL_PATH = "parallel_parking_logs/parallel_parking_ppo_model_best.zip"

os.makedirs(VIDEO_DIR, exist_ok=True)

# Create env with rendering
env = ParallelParkingEnv(render_mode="rgb_array")

# Wrap with video recorder
env = gym.wrappers.RecordVideo(
    env,
    video_folder=VIDEO_DIR,
    episode_trigger=lambda episode_id: True,  # record every episode
    name_prefix="parallel_parking_debug"
)

# Load model (or comment this out to test random actions)
model = PPO.load(MODEL_PATH)

obs, info = env.reset()
terminated = truncated = False
step = 0

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

print(f"Episode finished after {step} steps")
env.close()
