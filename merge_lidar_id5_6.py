import os

import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from tqdm import tqdm

LOG_DIR = "./merge_lidar_id5_6_logs/"
TENSORBOARD_DIR = "./merge_lidar_id5_6_tb/"
MODEL_NAME = "merge_lidar_id5_6_model"
NUM_ENVS = 8
TRAIN_STEPS = 500_000
EVAL_EPISODES = 500
RUN_TUNING = False
TUNING_CSV_PATH = "merge_lidar_id5_6_tuning_results.csv"
LEARNING_CURVE_PATH = "merge_lidar_id5_learning_curve_tuned.png"
LEARNING_CURVE_TITLE = "Learning Curve - PPO - Merge Env (LidarObservation)"
PERFORMANCE_VIOLIN_PATH = "merge_lidar_id6_performance_test_tuned.png"
PERFORMANCE_VIOLIN_TITLE = (
    "Performance 500 Episodes - PPO - Merge Env (LidarObservation)"
)


def plot_learning_curve(log_dir, save_path, title):
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
    plt.plot(mean_curve, label="Mean 300-ep Moving Avg (across envs)")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def plot_performance_violin(test_rewards, save_path, title):
    plt.figure(figsize=(6, 6))
    plt.violinplot(test_rewards, showmeans=True)
    plt.title(title)
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def make_env(rank):
    def _init():
        env = gym.make(
            "merge-v0",
            config={"observation": {"type": "LidarObservation"}},
        )
        os.makedirs(LOG_DIR, exist_ok=True)
        env = Monitor(
            env,
            filename=os.path.join(LOG_DIR, f"monitor_{rank}.csv"),
            allow_early_resets=True,
        )
        return env

    return _init


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def evaluate(model, env, episodes):
    rewards = []
    for _ in tqdm(range(episodes), desc="Evaluating"):
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs = reset_out
            info = {}

        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)

            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            total_reward += float(np.mean(reward))
            done = bool(np.all(done))

        rewards.append(total_reward)

    return np.array(rewards)


def tuning(env_fn, trials=25, short_steps=100_000):
    possible = {
        "learning_rate": [1e-4, 2e-4, 3e-4, 5e-4, 1e-3],
        "ent_coef": [0.001, 0.003, 0.005, 0.007, 0.01],
        "clip_range": [0.05, 0.1, 0.15, 0.2],
        "n_steps": [128, 256, 512],
        "batch_size": [128, 256, 512],
        "n_epochs": [5, 10, 15],
        "gae_lambda": [0.9, 0.95, 0.98],
    }

    results = []

    for i in range(trials):
        params = {}
        for name, values in possible.items():
            params[name] = np.random.choice(values)

        print(f"\n[Trial {i + 1}/{trials}] {params}")

        tmp_env = SubprocVecEnv([env_fn(j) for j in range(NUM_ENVS)])
        tmp_env = VecNormalize(tmp_env, norm_obs=True, norm_reward=False)

        model = PPO(
            policy="MlpPolicy",
            env=tmp_env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=0.99,
            gae_lambda=params["gae_lambda"],
            ent_coef=params["ent_coef"],
            clip_range=params["clip_range"],
            verbose=0,
            device="cpu",
        )

        model.learn(total_timesteps=short_steps, progress_bar=True)

        eval_env = env_fn(999)()
        episode_rewards = evaluate(model, eval_env, 50)

        mean_reward = float(np.mean(episode_rewards))
        eval_env.close()
        tmp_env.close()

        results.append((params, mean_reward))
        print(f"Avg reward (50 eps): {mean_reward:.3f}")

    results.sort(key=lambda x: x[1], reverse=True)
    best_params, best_reward = results[0]

    rows = []
    for i, (params, mean_reward) in enumerate(results):
        row = {"trial": i + 1, "mean_reward": mean_reward}
        for name, value in params.items():
            row[name] = value
        rows.append(row)

    pd.DataFrame(rows).to_csv(TUNING_CSV_PATH, index=False)

    print(f"\nBest Config {best_params} (mean reward={best_reward:.3f})")
    print(f"Results saved to {TUNING_CSV_PATH}\n")

    return best_params


def main():
    if RUN_TUNING:
        best = tuning(make_env, trials=20, short_steps=100_000)
    else:
        best = {
            "learning_rate": 0.0001,
            "ent_coef": 0.007,
            "clip_range": 0.1,
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 15,
            "gae_lambda": 0.98,
        }
        print("\nSkipping tuning parameters:")
        print(best)

    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    eval_env = SubprocVecEnv([make_env(-1)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(best["learning_rate"]),
        n_steps=best["n_steps"],
        batch_size=best["batch_size"],
        n_epochs=best["n_epochs"],
        gamma=0.99,
        gae_lambda=best["gae_lambda"],
        ent_coef=best["ent_coef"],
        clip_range=best["clip_range"],
        verbose=1,
        device="cpu",
        tensorboard_log=TENSORBOARD_DIR,
    )

    print(f"Training for {TRAIN_STEPS} timesteps.")
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)
    model.save(os.path.join(LOG_DIR, MODEL_NAME))
    train_env.save(os.path.join(LOG_DIR, "vecnormalize.pkl"))
    print("Training finished.")

    plot_learning_curve(LOG_DIR, LEARNING_CURVE_PATH, LEARNING_CURVE_TITLE)

    print(f"\nEvaluating on {EVAL_EPISODES} episodes")

    eval_env = VecNormalize.load(os.path.join(LOG_DIR, "vecnormalize.pkl"), eval_env)
    eval_env.training = False

    test_rewards = evaluate(model, eval_env, episodes=EVAL_EPISODES)

    print("\nSummary:")
    print(f"Mean reward: {np.mean(test_rewards):.3f}")

    plot_performance_violin(
        test_rewards, PERFORMANCE_VIOLIN_PATH, PERFORMANCE_VIOLIN_TITLE
    )


if __name__ == "__main__":
    main()
