import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

print(env.action_space)  # Discrete(3) : Accel left / Accel right / cease
print(env.observation_space)  # Box(2,) : CarPos / Velocity
print(env.observation_space.high)  # rightest position / highest positive velocity
print(env.observation_space.low)  # leftest position / highest negative velocity

num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
print(num_states)

num_states = np.round(num_states, 0).astype(int) + 1
print(num_states)
print(env.action_space.n)

Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))
print(Q.shape)