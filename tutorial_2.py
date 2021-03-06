import gym
# import keras
import random
import math
import numpy as np
from collections import deque  # some kind of faster list object


def play():
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                print(f"Episode finished after {t+1} timesteps")
                break


n_episodes = 1000
n_win_ticks = 195
max_env_steps = None

gamma = 1.0  # Discount Factor (influence the choice of best immediate reward vs long therm
epsilon = 1.0  # Exploration / supposed to have some kind of diminishing value
epsilon_min = .01
epsilon_decay = .995
alpha = .01  # learning rate
alpha_decay = .01

#  Gym parameters
batch_size = 64
monitor = False
quiet = False

memory = deque(maxlen=100000)
env = gym.make('CartPole-v1')
if max_env_steps is not None:
    env.max_episode_steps = max_env_steps


####  Building the neural network  ####
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# model definition
model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(2, activation='relu'))
model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))


# defining necessary function

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(model.predict(state))


# return the current value of epsilon or the minimum value if the decayed one is inferior to the set minimum
def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0 - math.log10(t+1)*epsilon_decay))


def preprocess_state(state):
    return np.reshape(state, [1, 4])  # put it into a column


def replay(batch_size, epsilon):
    x_batch, y_batch = [], []
    minibatch = random.sample(memory, min(len(memory), batch_size))

    for state, action, reward, next_state, done in minibatch:
        y_target = model.predict(state)
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


# define run function

def run():
    scores = deque(maxlen=100)
    for e in range(n_episodes):
        state = preprocess_state(env.reset())
        done = False
        i = 0
        while not done:
            action = choose_action(state, get_epsilon(e))
            next_state, reward, done, _ = env.step(action)
            env.render()
            next_state = preprocess_state(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1
        scores.append(i)
        mean_score = np.mean(scores)

        if mean_score >= n_win_ticks and e >= 100:
            if not quiet: print(f'Ran {e} episodes. Solved after {e-100} trails')
            return e - 100
        if e % 20 == 0 and not quiet and e != 0:
            print(f'[episode {e}] - Mean survival time over last 20 episodes was {mean_score} ticks')

        replay(batch_size, get_epsilon(e))

    if not quiet: print(f'Did not solve after {e} episodes')
    return e

# https://youtu.be/C2rpcQyv5bk?t=1886


run()