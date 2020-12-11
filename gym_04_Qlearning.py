import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
env.reset()
state_low = [-4.8, -10.0, -4.2, -10.0]
print(env.action_space)  # Discrete(3) : Accel left / Accel right / cease
print(env.observation_space)  # Box(2,) : CarPos / Velocity
print(env.observation_space.high)  # rightest position / highest positive velocity
print(env.observation_space.low)  # leftest position / highest negative velocity


def adjust_state(state):
    # Discretize state / adj for adjusted
    state_adj = (state - state_low) * np.array([10, 10, 10, 10])
    state_adj = np.round(state_adj, 0).astype(int)
    return state_adj


def q_learning(env, learning, discount, epsilon, min_eps, episodes):
    num_states = [97, 201, 85, 201]
    Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1],
                                                num_states[2], num_states[3],
                                                env.action_space.n))
    # Initialize variable to track rewards
    reward_list = []
    ave_reward_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps) / episodes

    for i in range(episodes):
        # Initialize parameters
        done = False
        y = 0
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state_adj = adjust_state(state)

        while done != True:
            y += 1
            # Render env for last five eps
            if i >= (episodes - 20):
                env.render()

            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize state2
            state2_adj = adjust_state(state2)

            # Allow for terminal states // .5 on env_space[0] represent the flag position
            if done and y == 195:
                Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action] = reward

            # adjust Q value for current state
            else:
                '''work on this, it's complicated but far from non-understandable'''
                delta = learning * (reward + discount
                                    * np.max(Q[state2_adj[0], state2_adj[1], state2_adj[2], state2_adj[3]])
                                    - Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action])
                Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action] += delta

            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        # Track rewards
        reward_list.append(tot_reward)

        if (i + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            print(f'Episode {i + 1} Average Reward: {ave_reward}')

    env.close()

    return ave_reward_list


rewards = q_learning(env, .05, .9, .8, 0, 300000)

# Plot Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')
plt.close()
