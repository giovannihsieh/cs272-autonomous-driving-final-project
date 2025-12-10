"""
Custom Highway Environment: Safe Lane Changing Highway
This environment encourages safe and efficient lane changing behavior.

Key modifications:
1. Custom reward function emphasizing:
   - Safe following distance
   - Minimal unnecessary lane changes
   - Right-lane discipline
   - Smooth speed control
2. Modified road configuration:
   - 4 lanes for more complex scenarios
   - Higher vehicle density
   - Varied vehicle speeds

Team: [Your Team ID]
Based on highway-env: https://github.com/Farama-Foundation/HighwayEnv
"""

import gymnasium as gym
from gymnasium.envs.registration import register
import highway_env
from highway_env import utils
from highway_env.envs import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
import numpy as np


class SafeLaneChangeHighwayEnv(HighwayEnv):
    """
    Custom highway environment focusing on safe lane changing behavior.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.last_lane = None
        self.lane_change_count = 0
        self.time_in_left_lanes = 0

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "LidarObservation",
                "cells": 64,
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,  # More lanes than standard highway
            "vehicles_count": 40,  # Higher density
            "duration": 50,  # Longer episodes
            "vehicles_density": 2.0,  # Denser traffic
            "collision_reward": -5.0,  # Severe penalty for collision
            "right_lane_reward": 0.2,  # Reward for using right lanes
            "high_speed_reward": 0.6,  # Moderate speed reward
            "lane_change_penalty": -0.1,  # Small penalty for lane changes
            "safe_distance_reward": 0.3,  # Reward for safe following distance
            "speed_limit": 30,  # m/s
            "policy_frequency": 2,  # Hz
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        })
        return config

    def _reward(self, action):
        """
        Custom reward function encouraging safe lane changing.

        Rewards:
        - Speed (scaled)
        - Right lane usage
        - Safe following distance

        Penalties:
        - Collisions (severe)
        - Frequent lane changes
        - Staying in left lanes unnecessarily
        """
        reward = 0

        # Get the ego vehicle
        vehicle = self.vehicle

        # 1. Speed reward (scaled by config)
        speed_reward = self.config["high_speed_reward"] * (
            vehicle.speed / vehicle.target_speed
        )
        reward += speed_reward

        # 2. Collision penalty
        if vehicle.crashed:
            reward += self.config["collision_reward"]
            return reward  # End episode

        # 3. Right lane discipline reward
        # Reward for staying in rightmost lanes when possible
        current_lane = vehicle.lane_index[2] if hasattr(vehicle, 'lane_index') else 0

        if hasattr(vehicle, 'lane_index'):
            # Lane 0 is rightmost, lane 3 is leftmost
            if current_lane == 0:  # Rightmost lane
                reward += self.config["right_lane_reward"] * 1.5
            elif current_lane == 1:
                reward += self.config["right_lane_reward"] * 1.0
            elif current_lane == 2:
                reward += self.config["right_lane_reward"] * 0.3
            # No reward for leftmost lane (3)

        # 4. Lane change penalty
        # Penalize frequent lane changes
        if self.last_lane is not None and current_lane != self.last_lane:
            reward += self.config["lane_change_penalty"]
            self.lane_change_count += 1

        self.last_lane = current_lane

        # 5. Safe distance reward
        # Reward maintaining safe distance from front vehicle
        front_vehicle, front_distance = self._get_front_vehicle()
        if front_vehicle is not None:
            safe_distance = vehicle.speed * 2  # 2 seconds rule
            if front_distance > safe_distance:
                reward += self.config["safe_distance_reward"]
            elif front_distance < safe_distance * 0.5:
                # Too close, small penalty
                reward -= 0.2

        return reward

    def _get_front_vehicle(self):
        """
        Get the vehicle directly in front in the same lane.
        Returns: (vehicle, distance) or (None, None)
        """
        vehicle = self.vehicle
        front_vehicle = None
        min_distance = float('inf')

        if not hasattr(vehicle, 'lane_index'):
            return None, None

        for v in self.road.vehicles:
            if v is vehicle:
                continue

            # Check if in same lane
            if hasattr(v, 'lane_index') and v.lane_index == vehicle.lane_index:
                # Check if in front
                if v.position[0] > vehicle.position[0]:
                    distance = v.position[0] - vehicle.position[0]
                    if distance < min_distance:
                        min_distance = distance
                        front_vehicle = v

        if front_vehicle is None:
            return None, None

        return front_vehicle, min_distance

    def _is_terminated(self):
        """Episode terminates on collision."""
        return self.vehicle.crashed

    def _is_truncated(self):
        """Episode truncates when duration exceeded."""
        return self.time >= self.config["duration"]

    def reset(self, **kwargs):
        """Reset environment and tracking variables."""
        self.last_lane = None
        self.lane_change_count = 0
        self.time_in_left_lanes = 0
        return super().reset(**kwargs)


# Register the custom environment
register(
    id='safe-lane-change-highway-v0',
    entry_point='custom_env:SafeLaneChangeHighwayEnv',
)


if __name__ == "__main__":
    # Test the custom environment
    print("Testing custom environment...")
    env = gym.make('safe-lane-change-highway-v0')

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Run a few steps
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: Reward = {reward:.3f}, Done = {done}, Truncated = {truncated}")

        if done or truncated:
            obs, info = env.reset()
            print("Episode reset")

    env.close()
    print("Custom environment test complete!")
