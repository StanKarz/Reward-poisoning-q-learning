import numpy as np


class AdversaryAgent:
    def __init__(self, episode, step, total_eps):
        self.episode = episode
        self.step = step
        self.total_eps = total_eps
        self.n1 = 6473
        self.n2 = 7079
        self.budget = total_eps * 35

    def get_noise(self, mu, sigma):
        seed = (self.episode * self.n1) + (self.step * self.n2)
        np.random.seed(seed)
        noise = np.random.normal(mu, sigma)
        if self.budget > 0:
            self.budget -= abs(noise)
            return noise
        else:
            return 0


