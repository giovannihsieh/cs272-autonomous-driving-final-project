import os
from typing import Callable
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def make_env(rank: int):
    def _init():
        env = gym.make(
            "highway-v0",
            config={
                "observation": {
                    "type": "GrayscaleObservation",
                    "observation_shape": (84, 84),
                    "stack_size": 4,
                    "weights": [0.2989, 0.5870, 0.1140],
                    "scaling": 1.75,
                }
            },
        )
        os.makedirs("./highway_grayscale_logs/", exist_ok=True)
        env = Monitor(env, filename=f"./highway_grayscale_logs/monitor_{rank}.csv")
        return env
    return _init


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    num_envs = 8
    train_env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = DummyVecEnv([make_env(-1)])

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=linear_schedule(3e-4),
        n_steps=1024, #increase from 256 to 1024 for more stable learning
        batch_size=512, #increase from 256
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.1,
        verbose=1,
        device="cuda", # use gpu instead of cpu
        tensorboard_log="./highway_grayscale_tensorboard/",
    )

    log_dir = "./highway_grayscale_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("Training PPO 500k steps Highway-v0 GrayscaleObservation.")
    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save(os.path.join(log_dir, "highway_grayscale"))
    print("Training finished.")


    all_curves = []

    for filename in os.listdir(log_dir):
        if filename.startswith("monitor_") and not filename.startswith("monitor_-1"):
            path = os.path.join(log_dir, filename)
            df = pd.read_csv(path, skiprows=1)
            df = df.sort_values(by="t")
            roll = df["r"].rolling(300, min_periods=1).mean().to_numpy()
            all_curves.append(roll)

    max_len = max(len(c) for c in all_curves)
    padded = np.full((len(all_curves), max_len), np.nan)
    for i, arr in enumerate(all_curves):
        padded[i, : len(arr)] = arr

    mean_curve = np.nanmean(padded, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_curve, label="Mean 300‑ep Moving Avg (across 8 envs)")
    plt.title("Learning Curve - PPO - Highway Env (GrayscaleObservation)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("highway_grayscale_learning_curve.png")
    plt.show()

    print("\nEvaluating on 500 episodes")
    def evaluate_agent(model, env, episodes=500):
        rewards = []
        for _ in tqdm(range(episodes)):
            obs = env.reset()
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_out = env.step(action)
                if len(step_out) == 4:
                    obs, reward, done, info = step_out
                else:
                    obs, reward, terminated, truncated, info = step_out
                    done = terminated or truncated
                total += float(np.mean(reward))
                if isinstance(done, (list, np.ndarray)) and np.all(done):
                    done = True
            rewards.append(total)
        return np.array(rewards)

    test_rewards = evaluate_agent(model, eval_env, episodes=500)
    print("\nSummary:")
    print(f"Mean reward: {np.mean(test_rewards):.3f}")

    plt.figure(figsize=(6, 6))
    plt.violinplot(test_rewards, showmeans=True)
    plt.title("Performance Test – Highway (GrayscaleObservation, Tuned)")
    plt.ylabel("Total Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("highway_grayscale_performance_test.png", dpi=300)
    plt.show()

    print("\nRecording videos")
    video_folder = "videos/highway_grayscale/"
    os.makedirs(video_folder, exist_ok=True)
    video_env = gym.make(
        "highway-v0",
        render_mode="rgb_array",
        config={
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (84, 84),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75,
            }
        },
    )
    video_env = RecordVideo(
        video_env,
        video_folder=video_folder,
        name_prefix="highway_grayscale_tuned_run",
        episode_trigger=lambda eid: True,
    )
    video_env = VecNormalize.load(
        os.path.join(log_dir, "vecnormalize.pkl"),
        video_env
    )
    video_env.training = False
    video_env.norm_reward = False

    model = PPO.load(os.path.join(log_dir, "highway_grayscale"), env=video_env)

    for ep in tqdm(range(5)):
        obs, _ = video_env.reset()
        done = ter = tru = False
        total = 0
        while not (ter or tru or done):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, ter, tru, _ = video_env.step(action)
            total += reward
        print(f"Episode {ep + 1}/5 Reward = {total:.2f}")

    video_env.close()