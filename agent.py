import random


class QLearningAgent:
    def __init__(self, env, epsilon, learning_rate, discount_factor):
        self.epsilon = epsilon
        self.gamma = discount_factor
        self.lr = learning_rate
        self.q_table = {}
        self.n1 = 6473
        self.n2 = 7079

    def take_action(self, state, episode, step):
        # Generate random seed using two large primes, current episode and step
        seed = (episode * self.n1) + (step * self.n2)
        random.seed(seed)
        # Sample a random action
        random_action = random.randint(0, 2)
        # Generate random number from 0 to 1, to be used for comparison with epsilon value
        random_num = random.uniform(0, 1)

        # Retrieve max Q-value, given the current state
        max_q_val = max(self.q_table[state, 0], self.q_table[state, 1], self.q_table[state, 2])

        # Find the action associated with the highest Q-value
        for key, value in self.q_table.items():
            if key[0] == state and value == max_q_val:
                max_action = key[1]

        # Epsilon greedy policy
        if random_num > self.epsilon:
            action = max_action  # Action associated with max Q-value
        else:
            action = random_action
        return action

    def update_q_table(self, state, action, reward, next_state):
        # Retrieve max Q-value associated with the next state
        max_q_val = max(self.q_table[next_state, 0], self.q_table[next_state, 1], self.q_table[next_state, 2])
        # Q-table update
        self.q_table[state, action] = self.q_table[state, action] + self.lr * (reward + self.gamma * max_q_val) - self.q_table[state, action]
