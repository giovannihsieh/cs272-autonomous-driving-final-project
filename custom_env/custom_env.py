import gymnasium
import custom_env.emergency_env
import highway_env
import matplotlib
from matplotlib import pyplot as plt
import time


env = gymnasium.make('EmergencyHighwayEnv-v0', render_mode='human')

env.reset()

env_unwrapped = env.unwrapped
# plot starting positions
x_positions = []
y_positions = []
colors = []

for v in env_unwrapped.road.vehicles:
    x_positions.append(v.position[0])
    y_positions.append(v.lane_index[2])

    if v is env_unwrapped.vehicle:
        colors.append('cyan') # for ego vehicle
    elif getattr(v, "is_emergency", False):
        colors.append('red') # for emergency vehicle
    else:
        colors.append('gray') # for regular vehicle

# plt.figure(figsize=(10, 4))
# plt.scatter(x_positions, y_positions, c=colors, s=200)
# plt.xlabel("Horizontal position (x)")
# plt.ylabel("Lane index")
# plt.title("Starting spawn locations of vehicles")
# num_lanes = max(v.lane_index[2] for v in env_unwrapped.road.vehicles) + 1
# plt.yticks(range(num_lanes))
# plt.grid(True)
# plt.show()

# Keep simulation running until ego crashes
step = 0
while not env_unwrapped.vehicle.crashed:
    action = env_unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    step += 1

print("ego crashed.")