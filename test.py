import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make('highway-fast-v0')
observation = env.reset()
print("\nobservation.shape :", observation.shape)
for _ in range(20):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
plt.imshow(env.render(mode="rgb_array"))
plt.show()