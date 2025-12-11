import os
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from parallel_parking_env import ParallelParkingEnv
from merge_lidar_id5_6 import (
    evaluate,
    linear_schedule,
    plot_learning_curve,
    plot_performance_violin,
    tuning,
)

LOG_DIR = "./parallel_parking_logs/"
TENSORBOARD_DIR = "./parallel_parking_tb/"
MODEL_NAME = "parallel_parking_ppo_model"

NUM_ENVS = 8
TRAIN_STEPS = 1000_000
EVAL_EPISODES = 500

RUN_TUNING = False

LEARNING_CURVE_PATH = "parallel_parking_learning_curve.png"
LEARNING_CURVE_TITLE = "Learning Curve - PPO - Parallel Parking"
PERFORMANCE_VIOLIN_PATH = "parallel_parking_performance.png"
PERFORMANCE_VIOLIN_TITLE = "Performance (500 Episodes) - PPO - Parallel Parking"

# limit steering and acceleration to try to prevent early crashes so model can learn
class DynamicSafeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_steer=0.1, max_accel=0.2, safe_steps=10):
        super().__init__(env)
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.safe_steps = safe_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def action(self, action):
        self.current_step += 1
        # Scale factor: starts at 0, linearly increases to 1 over safe_steps
        scale = min(1.0, self.current_step / self.safe_steps)
        # Gradually relax the clipping range
        steer_limit = self.max_steer * scale
        accel_limit = self.max_accel * scale
        # Clip the action
        return np.clip(action, [-steer_limit, -accel_limit], [steer_limit, accel_limit])

# increase reward when closer to goal, subtract when move further
class DistanceShapingWrapper(gym.Wrapper):
    def __init__(self, env, k=1.0, gamma=0.99):
        super().__init__(env)
        self.k = k
        self.gamma = gamma
        self.prev_potential = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = self._potential()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_potential = self._potential()
        shaping = self.gamma * new_potential - self.prev_potential
        self.prev_potential = new_potential
        return obs, reward + shaping, terminated, truncated, info

    def _potential(self):
        # Access unwrapped env’s goal and vehicle
        env = self.env.unwrapped
        p = env.vehicle.position
        d = np.linalg.norm(p - env.goal_position)
        return -self.k * d

# try to prevent crashes
class SurvivalBonusWrapper(gym.RewardWrapper):
    def __init__(self, env, per_step_bonus=0.1):
        super().__init__(env)
        self.per_step_bonus = per_step_bonus

    def reward(self, reward):
        return reward + self.per_step_bonus

def make_env(rank: int):
    def _init():
        env = ParallelParkingEnv(render_mode=None)
        # wrappers
        #env = DynamicSafeActionWrapper(env, max_steer=0.3, max_accel=0.6, safe_steps=10)
        #env = DistanceShapingWrapper(env, k=0.5, gamma=0.99)
        #env = SurvivalBonusWrapper(env, per_step_bonus=0.05)
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
            "learning_rate": 3e-4,
            "ent_coef": 0.02,
            "clip_range": 0.2,
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 10,
            "gae_lambda": 0.95,
        }
        print("\nSkipping tuning parameters:")
        print(best)

    # Vectorized environments
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env(-1)])

    # Normalize
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    # PPO model
    model = PPO(
        policy="MultiInputPolicy",
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
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        ),
    )

    print(f"\nTraining for {TRAIN_STEPS} timesteps.")
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)

    model.save(os.path.join(LOG_DIR, MODEL_NAME))
    train_env.save(os.path.join(LOG_DIR, "vecnormalize.pkl"))
    print("Training finished.")

    plot_learning_curve(LOG_DIR, LEARNING_CURVE_PATH, LEARNING_CURVE_TITLE)

    print(f"\nEvaluating on {EVAL_EPISODES} episodes")

    eval_env = VecNormalize.load(
        os.path.join(LOG_DIR, "vecnormalize.pkl"), eval_env
    )
    eval_env.training = False

    test_rewards = evaluate(model, eval_env, episodes=EVAL_EPISODES)

    print("\nSummary:")
    print(f"Mean reward: {np.mean(test_rewards):.3f}")

    plot_performance_violin(
        test_rewards,
        PERFORMANCE_VIOLIN_PATH,
        PERFORMANCE_VIOLIN_TITLE,
    )


if __name__ == "__main__":
    main()
