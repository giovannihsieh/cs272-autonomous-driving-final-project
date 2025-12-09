import os
from typing import Callable
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def make_env(rank: int):
    def _init():
        env = gym.make(
            "highway-v0",
            config={"observation": {"type": "LidarObservation"}},
        )
        os.makedirs("./highway_lidar_logs/", exist_ok=True)
        env = Monitor(
            env,
            filename=f"./highway_lidar_logs/monitor_{rank}.csv",
            allow_early_resets=True,
        )
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

    train_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = SubprocVecEnv([make_env(-1)])

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(3e-4),
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.1,
        verbose=1,
        device="cpu",
        tensorboard_log="./highway_lidar_tensorboard/",
    )

    log_dir = "./highway_lidar_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("Training PPO 200k steps Highway‑v0 Lidar.")
    model.learn(total_timesteps=200_000, progress_bar=True)
    model.save(os.path.join(log_dir, "highway_lidar"))
    train_env.save(os.path.join(log_dir, "vecnormalize.pkl"))
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
    plt.title("Learning Curve - PPO - Highway Env (LidarObservation)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("highway_lidar_learning_curve.png")
    plt.show()

    print("\nEvaluating on 500 episodes")

    def evaluate_agent(model, env, episodes=500):
        rewards = []
        for _ in tqdm(range(episodes)):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
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

    eval_env = VecNormalize.load(os.path.join(log_dir, "vecnormalize.pkl"), eval_env)
    eval_env.training = False

    test_rewards = evaluate_agent(model, eval_env, episodes=500)
    print("\nSummary:")
    print(f"Mean reward: {np.mean(test_rewards):.3f}")

    plt.figure(figsize=(6, 6))
    plt.violinplot(test_rewards, showmeans=True)
    plt.title("Performance 500 episodes - PPO - Highway Env (LidarObservation)")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig("highway_lidar_performance_test.png")
    plt.show()

    print("\nRecording videos")
    video_folder = "videos/highway_lidar/"
    os.makedirs(video_folder, exist_ok=True)

    video_env = gym.make(
        "highway-v0",
        render_mode="rgb_array",
        config={"observation": {"type": "LidarObservation"}},
    )

    video_env = RecordVideo(
        video_env,
        video_folder=video_folder,
        name_prefix="highway_lidar_tuned_run",
        episode_trigger=lambda eid: True,
    )

    model = PPO.load(os.path.join(log_dir, "highway_lidar"), env=video_env)

    for ep in tqdm(range(5)):
        obs, info = video_env.reset()
        done = truncated = False
        total = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = video_env.step(action)
            total += reward
        print(f"Episode {ep + 1}/5 Reward = {total:.2f}")

    video_env.close()
