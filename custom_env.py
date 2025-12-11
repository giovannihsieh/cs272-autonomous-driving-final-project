
from parallel_parking_env import ParallelParkingEnv
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
from time import sleep
from IPython import display
import numpy as np


env = ParallelParkingEnv(render_mode="rgb_array")

frames = []
obs, info = env.reset()
frame = env.render()
frames.append(np.array(frame, dtype=np.uint8))
for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    frames.append(np.array(frame, dtype=np.uint8))

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

env.close()

for frame in frames:
    plt.imshow(frame)
    plt.axis("off")
    display.display(plt.gcf())
    sleep(2)
    display.clear_output(wait=True)

