import os

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from merge_lidar_id5_6 import (
    evaluate,
    linear_schedule,
    plot_learning_curve,
    plot_performance_violin,
    tuning,
)

LOG_DIR = "./highway_grayscale_id3_4_logs/"
TENSORBOARD_DIR = "./highway_grayscale_id3_4_tb/"
MODEL_NAME = "highway_grayscale_id3_4_model"
NUM_ENVS = 8
TRAIN_STEPS = 500_000
EVAL_EPISODES = 500
RUN_TUNING = False
TUNING_CSV_PATH = "highway_grayscale_id3_4_tuning_results.csv"
LEARNING_CURVE_PATH = "highway_grayscale_id3_learning_curve_tuned.png"
LEARNING_CURVE_TITLE = "Learning Curve - PPO - Highway Env (GrayscaleObservation)"
PERFORMANCE_VIOLIN_PATH = "highway_grayscale_id4_performance_test_tuned.png"
PERFORMANCE_VIOLIN_TITLE = (
    "Performance 500 Episodes - PPO - Highway Env (GrayscaleObservation)"
)


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
        os.makedirs(LOG_DIR, exist_ok=True)
        env = Monitor(env, filename=os.path.join(LOG_DIR, f"monitor_{rank}.csv"))
        return env

    return _init


def main():
    if RUN_TUNING:
        best = tuning(make_env, trials=20, short_steps=100_000)
    else:
        best = {
            "learning_rate": 0.0001,
            "ent_coef": 0.005,
            "clip_range": 0.1,
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 10,
            "gae_lambda": 0.95,
        }
        print("\nSkipping tuning parameters:")
        print(best)

    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env(-1)])

    model = PPO(
        policy="CnnPolicy",
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

    print(f"Training for {TRAIN_STEPS} steps.")
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)
    model.save(os.path.join(LOG_DIR, MODEL_NAME))
    print("Training finished.")

    plot_learning_curve(LOG_DIR, LEARNING_CURVE_PATH, LEARNING_CURVE_TITLE)

    print(f"\nEvaluating on {EVAL_EPISODES} episodes")

    test_rewards = evaluate(model, eval_env, episodes=EVAL_EPISODES)

    print("\nSummary:")
    print(f"Mean reward: {np.mean(test_rewards):.3f}")

    plot_performance_violin(
        test_rewards, PERFORMANCE_VIOLIN_PATH, PERFORMANCE_VIOLIN_TITLE
    )


if __name__ == "__main__":
    main()
