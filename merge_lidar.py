import os
import glob
from typing import Callable
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


def make_env(rank: int):
    def _init():
        env = gym.make("merge-v0", config={"observation": {"type": "LidarObservation"}})
        os.makedirs("./merge_lidar_logs/", exist_ok=True)
        monitor_file = f"./merge_lidar_logs/monitor_{rank}.csv"
        env = Monitor(env, filename=monitor_file, allow_early_resets=True)
        return env

    return _init


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    # cpu/threads
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
        tensorboard_log="./merge_lidar_tensorboard/",
    )

    log_dir = "./merge_lidar_logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("Starting PPO training on Merge-v0 (LidarObservation, parallelized)...")
    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save(os.path.join(log_dir, "merge_lidar"))
    train_env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print("Training finished, model and normalization stats saved.")

    print("Merging parallel monitor logs...")
    csv_files = glob.glob(os.path.join("./merge_lidar_logs/", "monitor_*.csv"))
    dfs = [pd.read_csv(f, skiprows=1) for f in csv_files if os.path.getsize(f) > 0]
    data = pd.concat(dfs, ignore_index=True)
    data.sort_index(inplace=True)
    merged_csv_path = os.path.join("./merge_lidar_logs/", "merged_monitor.csv")
    data.to_csv(merged_csv_path, index=False)

    try:
        rewards = data["r"]
        rolling_mean = rewards.rolling(window=100).mean()

        plt.figure(figsize=(9, 5))
        plt.plot(rolling_mean, label="100-ep Moving Average")
        plt.title("Learning Curve – Merge (LidarObservation, Parallel)")
        plt.xlabel("Episode")
        plt.ylabel("Mean Episode Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("ID_5_merge_lidar_learning_curve_tuned.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"Could not plot learning curve: {e}")

    print("\nEvaluating the tuned PPO model (500 episodes)...")

    def evaluate_agent(model, env, episodes=500):
        rewards = []
        for ep in tqdm(range(episodes)):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_out = env.step(action)
                if len(step_out) == 4:
                    obs, reward, done, info = step_out
                else:
                    obs, reward, terminated, truncated, info = step_out
                    done = terminated or truncated
                total_reward += float(np.mean(reward))
                if isinstance(done, (list, np.ndarray)) and np.all(done):
                    done = True
            rewards.append(total_reward)
        return np.array(rewards)

    eval_env = VecNormalize.load(os.path.join(log_dir, "vecnormalize.pkl"), eval_env)
    eval_env.training = False

    test_rewards = evaluate_agent(model, eval_env, episodes=500)

    print("\nEvaluation summary over 500 episodes:")
    print(f"Mean reward: {np.mean(test_rewards):.3f}")
    print(f"Std. dev.: {np.std(test_rewards):.3f}")
    print(f"Win rate (>0 reward): {np.mean(test_rewards > 0) * 100:.1f}%")

    plt.figure(figsize=(6, 6))
    plt.violinplot(test_rewards, showmeans=True)
    plt.title("Performance Test – Merge (LidarObservation, Tuned)")
    plt.ylabel("Total Episode Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ID_6_merge_lidar_performance_test_tuned.png", dpi=300)
    plt.show()

    print("All done with training, evaluation, and plot generation.")

    print("\nRecording demonstration videos of tuned PPO agent...")
    video_folder = "videos/merge_lidar/"
    os.makedirs(video_folder, exist_ok=True)

    video_env = gym.make(
        "merge-v0",
        render_mode="rgb_array",
        config={"observation": {"type": "LidarObservation"}},
    )

    video_env = RecordVideo(
        video_env,
        video_folder=video_folder,
        name_prefix="merge_lidar_tuned_run",
        episode_trigger=lambda episode_id: True,
    )
    print("Video environment ready (default merge, LidarObservation only).")

    model = PPO.load(os.path.join(log_dir, "merge_lidar"), env=video_env)

    num_episodes = 5
    for ep in tqdm(range(num_episodes)):
        obs, info = video_env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = video_env.step(action)
            total_reward += reward
        print(f"Episode {ep + 1}/{num_episodes} finished. Reward = {total_reward:.2f}")

    video_env.close()
    print(f"\n All demonstration videos saved to '{video_folder}'.")
    print(
        "\nTask 1 - Merge (LidarObservation, Tuned) complete — environment unchanged."
    )
