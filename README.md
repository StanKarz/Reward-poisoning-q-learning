# Investigating the effectiveness of reward poisoning methods with Q-learning in a grid environment

In this project, I will be investigating how much interference is necessary to disrupt the learning of a Q-learning reinforcement learning agent in a 
grid environment. Assuming an adversarial scenario, I will also explore if there is a better way to inject noise other than simple poisoning. 
The experiments were conducted using a simple 6x6 [mini-grid environment](https://github.com/Farama-Foundation/gym-minigrid) with a single goal state 
which provides a sparse reward if the agent reaches it. A small penalty is applied based on the number of steps required to reach the goal, 
the greater the number of steps the greater the penalty.


