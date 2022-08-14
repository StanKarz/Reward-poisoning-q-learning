import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import agent
import adversary

env = ImgObsWrapper(gym.make("MiniGrid-Empty-8x8-v0"))
env.reset()

# Agent hyperparameters
discount_factor = 0.95
learning_rate = 0.1
max_epsilon = 1.0

agent = agent.QLearningAgent(env=env, discount_factor=discount_factor, learning_rate=learning_rate, epsilon=max_epsilon)

# Adversary hyperparameters
episode = None
step = None
total_eps = 5000

adversary = adversary.AdversaryAgent(episode, step, total_eps)


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


for episode in range(total_episodes):
    state = env.reset()
    # Decay epsilon exponentially
    agent.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    actions_taken = []
    noise_added = 0
    total_rewards = 0
    reward_with_no_noise = 0

    for step in range(max_steps):
        state = to_tuple(state)

        # Provide episode and step data to adversary and sample noise value
        adversary.episode = episode
        adversary.step = step
        noise = adversary.get_noise(0, 0.75)

        for i in range(0, 3):
            state_action = state, i
            if state_action not in agent.q_table:
                agent.q_table[state_action] = 0

        action = agent.take_action(state, episode, step)
        next_state, reward, done, info = env.step(action)

        actions_taken.append(action)
        reward_with_no_noise += reward

        if done is True or step == 99:
            total_rewards += (reward + noise)
            noise_added += noise

            next_state = to_tuple(next_state)
            for i in range(0, 3):
                new_state_action = next_state, i
                if new_state_action not in agent.q_table:
                    agent.q_table[new_state_action] = 0

            agent.update_q_table(state, action, total_rewards, next_state)
            break

        else:
            total_rewards += reward
            next_state = to_tuple(next_state)
            for i in range(0, 3):
                new_state_action = next_state, i
                if new_state_action not in agent.q_table:
                    agent.q_table[new_state_action] = 0

            agent.update_q_table(state, action, total_rewards, next_state)
            state = next_state

            # New dict containing total reward and episode  is added to a list at the end of each episode

    rewards_dict['episode_number'] = episode
    rewards_dict['total_reward'] = total_rewards
    rewards_dict["reward_with_no_noise"] = reward_with_no_noise
    rewards_dict["noise_added"] = noise_added
    rewards_dict['steps'] = step
    rewards_dict["actions_taken"] = actions_taken

    rewards_array.append(dict(rewards_dict))
