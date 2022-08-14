import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import agent


# Training hyperparameters
discount_factor = 0.95
learning_rate = 0.1
max_epsilon = 1.0

# Setup environment
env = ImgObsWrapper(gym.make("MiniGrid-Empty-8x8-v0"))

env.reset()

agent = agent.QLearningAgent(env=env, discount_factor=discount_factor, learning_rate=learning_rate, epsilon=max_epsilon)

# Training hyperparameters
total_episodes = 5000
min_epsilon = 0.05
max_steps = 100
decay_rate = 0.0005

rewards_dict = {}
rewards_array = []


# Convert RGB state array to tuple
def to_tuple(array):
    try:
        return tuple(to_tuple(item) for item in array)
    except TypeError:
        return array


# Training hyperparameters
total_episodes = 5000
min_epsilon = 0.05
max_steps = 100
decay_rate = 0.0005

rewards_dict = {}
rewards_array = []

for episode in range(total_episodes):
    state = env.reset()
    # Decay epsilon exponentially
    agent.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    actions_taken = []
    total_rewards = 0

    for step in range(max_steps):
        state = to_tuple(state)

        # Check if state, action pair is in the Q-table if not create a key, value pair and set to 0
        for i in range(0, 3):
            state_action = state, i
            if state_action not in agent.q_table:
                agent.q_table[state_action] = 0

        action = agent.take_action(state, episode, step)
        next_state, reward, done, info = env.step(action)
        actions_taken.append(action)
        total_rewards += reward

        next_state = to_tuple(next_state)
        # Check if next state, action pair is in the Q-table if not create a key, value pair and set to 0
        for i in range(0, 3):
            new_state_action = next_state, i
            if new_state_action not in agent.q_table:
                agent.q_table[new_state_action] = 0

        agent.update_q_table(state, action, reward, next_state)
        state = next_state

        if done:
            break

    rewards_dict['episode_number'] = episode
    rewards_dict['total_reward'] = total_rewards
    rewards_dict['steps'] = step
    rewards_dict["actions_taken"] = actions_taken

    rewards_array.append(dict(rewards_dict))