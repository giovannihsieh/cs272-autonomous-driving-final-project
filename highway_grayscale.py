import gymnasium
import highway_env
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


class ChannelDepthLastWrapper(ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    stack, height, width = self.observation_space.shape
    low = np.transpose(self.observation_space.low, (1, 2, 0))
    high = np.transpose(self.observation_space.high, (1, 2, 0))
    self.observation_space = Box(
        low=low,
        high=high,
        shape=(height, width, stack),
        dtype=self.observation_space.dtype,
    )

  def observation(self, observation):
    return np.transpose(observation, (1, 2, 0))


def make_grayscale_env():
  config = {
      "observation": {
          "type": "GrayscaleObservation",
          "observation_shape": (128, 64),
          "stack_size": 4,
          "weights": [0.2989, 0.5870, 0.1140],
          "scaling": 1.75,
      },
      "policy_frequency": 2,
  }
  env = gymnasium.make("highway-v0", render_mode="rgb_array", config=config)
  env = ChannelDepthLastWrapper(env)
  return env


train_env = DummyVecEnv([make_grayscale_env])
train_env = VecTransposeImage(train_env)

model = DQN(
    "CnnPolicy",
    train_env,
    buffer_size=75_000,
    learning_rate=1e-4,
    learning_starts=10_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="highway_dqn/",
)

model.learn(total_timesteps=200_000)
model.save("highway_dqn/grayscale_model")

# Load and test the saved model in a non-vectorized env for rendering
eval_env = make_grayscale_env()
model = DQN.load("highway_dqn/grayscale_model")

while True:
  done = truncated = False
  obs, info = eval_env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    eval_env.render()