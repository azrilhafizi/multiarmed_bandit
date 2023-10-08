import numpy as np

class BernoulliBandit:
    def __init__(self, n_actions=5):
        self._probs = np.random.random(n_actions)

    @property
    def action_count(self):
        return len(self._probs)

    def pull(self, action):
        if np.any(np.random.random() > self._probs[action]):
            return 0.0
        return 1.0

    def optimal_reward(self):
        """
        Used for regret calculation
        """
        return np.max(self._probs)

    def action_value(self, action):
        """
        Used for regret calculation
        """
        return self._probs[action]

    def step(self):
        """ 
        Used in nonstationary version
        """
        pass

    def reset(self):
        """ 
        Used in nonstationary version
        """
        pass

class DriftingBandit(BernoulliBandit):
    def __init__(self, n_actions=5, gamma=0.01):
        super().__init__(n_actions)
        self._gamma = gamma
        self._successes = None
        self._failures = None
        self._steps = 0
        self.reset()

    def reset(self):
        self._successes = np.zeros(self.action_count) + 1.0
        self._failures = np.zeros(self.action_count) + 1.0
        self._steps = 0

    def step(self):
        action = np.random.randint(self.action_count)
        reward = self.pull(action)
        self._step(action, reward)

    def _step(self, action, reward):
        self._successes = self._successes * (1 - self._gamma) + self._gamma
        self._failures = self._failures * (1 - self._gamma) + self._gamma
        self._steps += 1
        self._successes[action] += reward
        self._failures[action] += 1.0 - reward
        self._probs = np.random.beta(self._successes, self._failures)