import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane
from highway_env.envs.common.observation import KinematicsGoalObservation


class ParallelParkingEnv(AbstractEnv):

    def __init__(self, config=None, render_mode=None):
        super().__init__(config=config, render_mode=render_mode)


    def default_config(self):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales":   [50,   10,  10,  10,     1,      1],
                "normalize": True
            },
            "action": {"type": "ContinuousAction"},

            "duration": 200,
            "collision_reward": -5,
            "success_reward": 10,
            "out_of_bounds_penalty": -5,
            "goal_tolerance": 0.4,
            "near_goal_position_reward": 2.0,
            "near_goal_reward": 3.0,
            "additional_alignment_reward":20.0
        })
        return config

    def _reset(self):

        net = RoadNetwork()
        lane_w = 4.0
        net.add_lane("a", "b", StraightLane([0, 0], [50, 0], width=lane_w))
        net.add_lane("a", "b", StraightLane([0, 4], [50, 4], width=lane_w))
        net.add_lane("a", "b", StraightLane([0, 8], [50, 8], width=lane_w))

        self.road = Road(network=net, np_random=self.np_random)
        self.road.vehicles = []

        parked_positions = [0, 24]
        for x in parked_positions:

            parked = Vehicle(self.road, position=np.array([float(x), 8.0]), heading=0, speed=0)
            parked.color = (255, 255, 255)
            self.road.vehicles.append(parked)

        ego = Vehicle(self.road, position=np.array([6.0, 4.0]), heading=0, speed=0)
        ego.color = (255, 0, 0)
        self.vehicle = ego
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

        self.goal_position = np.array([16.0, 8.0])
        self.goal_heading = 0.0

        ego.goal = type("Goal", (), {})()
        ego.goal.position = self.goal_position
        ego.goal.heading = self.goal_heading
        ego.goal.to_dict = lambda: {
            "x": self.goal_position[0],
            "y": self.goal_position[1],
            "vx": 0.0,
            "vy": 0.0,
            "cos_h": np.cos(self.goal_heading),
            "sin_h": np.sin(self.goal_heading)
        }


        cfg = self.config["observation"]
        self.observation_type = KinematicsGoalObservation(
            env=self,
            features=cfg["features"],
            scales=cfg["scales"],
            normalize=cfg["normalize"]
        )

        self.steps = 0

        return self.observation_type.observe()

    def _step(self, action):
        self.vehicle.act(action)
        self.road.step(1.0)
        self.steps += 1

        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        return self.observation_type.observe(), reward, terminated, truncated, {}

    def _reward(self, action):
        if self.vehicle.crashed:
            return self.config["collision_reward"]

        reward = 0.0
        p = self.vehicle.position
        distance = np.linalg.norm(p - self.goal_position)
        heading_diff = self._compute_heading_diff()

        reward -= 0.3 * np.clip(distance / 30.0, 0, 1)

        x, y = p
        if not (0 <= x <= 24):
            reward += self.config["out_of_bounds_penalty"]
        if not (0 <= y <= 10):
            reward += self.config["out_of_bounds_penalty"]

        if distance < self.config["goal_tolerance"]:
            alignment = 1 - heading_diff / np.pi
            reward += alignment * self.config["near_goal_position_reward"]

        if distance < self.config["goal_tolerance"] and heading_diff < np.deg2rad(15):
            reward += self.config["success_reward"]
        if distance < self.config["goal_tolerance"] and heading_diff < np.deg2rad(5):
            reward += self.config["additional_alignment_reward"]

        if 14 <= x <= 18 and 6 <= y <= 10:
            reward += self.config["near_goal_reward"]

        return reward

    def _is_terminated(self):
        distance = np.linalg.norm(self.vehicle.position - self.goal_position)
        heading_diff = self._compute_heading_diff()

        goal_reached = (
            distance < self.config["goal_tolerance"] and
            heading_diff < np.deg2rad(15)
        )

        return bool(self.vehicle.crashed or goal_reached)

    def render(self):
        return super().render()

    def _is_truncated(self):
        if self.steps >= self.config["duration"]:
            return True

        x, y = self.vehicle.position
        return not (0 <= x <= 24 and 0 <= y <= 10)


    def _compute_heading_diff(self):
        return abs(np.arctan2(
            np.sin(self.vehicle.heading - self.goal_heading),
            np.cos(self.vehicle.heading - self.goal_heading)
        ))



