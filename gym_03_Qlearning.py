import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.reset()


def adjust_state(state):
    # Discretize state / adj for adjusted
    state_adj = (state - env.observation_space.low) * np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)  # ex:[-0.4, .03] => [8, 10]
    return state_adj


#  Define Q-learning function
def q_learning(env, learning, discount, epsilon, min_eps, episodes):
    """Determine size of discrete tate space
       [-1.2  -0.07] low
       [0.6  0.07] high
       We will use those values as dimension to categorize the car location in the q-learning box
       ex : [-0.4, .03] => box corresponding of indices for [8, 10]
    """
    # [18.00000072 14.00000006]
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([10, 100])  # >> [18.00000072 14.00000006]
    num_states = np.round(num_states, 0).astype(int) + 1  # >> [19 15]

    # Initialize Q table
    # env.action_space.n return the number of action that our agent can make (here 3, left, cease, right)
    Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))

    # Initialize variable to track rewards
    reward_list = []
    ave_reward_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps) / (episodes / 2)

    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state_adj = adjust_state(state)

        while done != True:
            # Render env for last five eps
            if i >= (episodes - 20):
                env.render()

            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize state2
            state2_adj = adjust_state(state2)

            # Allow for terminal states // .5 on env_space[0] represent the flag position
            if done and state2[0] >= .5:
                Q[state_adj[0], state_adj[1], action] = reward

            # adjust Q value for current state
            else:
                '''work on this, it's complicated but far from non-understandable'''
                delta = learning*(reward + discount*np.max(Q[state2_adj[0], state2_adj[1]]) -
                                  Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        # Track rewards
        reward_list.append(tot_reward)

        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            print(f'Episode {i+1} Average Reward: {ave_reward}')

    env.close()

    return ave_reward_list


rewards = q_learning(env, .2, .9, .8, 0, 10000)

# Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')
plt.close()