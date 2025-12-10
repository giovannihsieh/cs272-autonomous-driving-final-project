import os

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from merge_lidar_id5_6 import (
    evaluate,
    linear_schedule,
    plot_learning_curve,
    plot_performance_violin,
    tuning,
)

LOG_DIR = "./intersection_lidar_id9_10_logs/"
TENSORBOARD_DIR = "./intersection_lidar_id9_10_tb/"
MODEL_NAME = "intersection_lidar_id9_10_model"
NUM_ENVS = 8
TRAIN_STEPS = 500_000
EVAL_EPISODES = 500
RUN_TUNING = False
TUNING_CSV_PATH = "intersection_lidar_id9_10_tuning_results.csv"
LEARNING_CURVE_PATH = "intersection_lidar_id9_learning_curve_tuned.png"
LEARNING_CURVE_TITLE = "Learning Curve - PPO - Intersection Env (LidarObservation)"
PERFORMANCE_VIOLIN_PATH = "intersection_lidar_id10_performance_test_tuned.png"
PERFORMANCE_VIOLIN_TITLE = (
    "Performance 500 Episodes - PPO - Intersection Env (LidarObservation)"
)

def make_env(rank):
    def _init():
        env = gym.make(
            "intersection-v0",
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
