import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import no_poisoning
import simple_poisoning

env = ImgObsWrapper(gym.make("MiniGrid-Empty-8x8-v0"))
env.reset()

max_steps = 100
eval_episodes = 100
env.reset()

eval_array = []
eval_dict = {}


def to_tuple(array):
    try:
        return tuple(to_tuple(item) for item in array)
    except TypeError:
        return array


for episode in range(eval_episodes):
    actions_taken = []
    state = env.reset()
    total_rewards = 0

    for step in range(max_steps):

        state = to_tuple(state)
        max_val = max(no_poisoning.agent_q_table[state, 0], no_poisoning.agent_q_table[state, 1], no_poisoning.agent_q_table[state, 2])
        for k, v in no_poisoning.agent.q_table.items():
            if k[0] == state and v == max_val:
                max_action = k[1]

        next_state, reward, done, info = env.step(max_action)
        actions_taken.append(max_action)
        total_rewards += reward
        state = next_state

        if done:
            break

    eval_dict['total_reward'] = total_rewards
    eval_dict['episode_number'] = episode
    eval_dict['steps'] = step
    eval_dict["actions_taken"] = actions_taken
    eval_array.append(dict(eval_dict))

