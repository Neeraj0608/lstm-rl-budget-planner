import gym
import numpy as np
from gym import spaces

class BudgetEnv(gym.Env):
    def __init__(self, income, savings, predicted_expense, ratios):
        super().__init__()

        self.income = float(income)
        self.savings = float(savings)
        self.predicted_expense = float(predicted_expense)

        self.available_budget = max(self.income - self.savings, 1.0)

        self.ratios = np.array(ratios, dtype=np.float32)
        self.n = len(self.ratios)

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n + 2,), dtype=np.float32
        )

    def reset(self):
        return self._get_state()

    def _get_state(self):
        return np.concatenate([
            self.ratios,
            [self.predicted_expense / self.income],
            [self.available_budget / self.income]
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)

        total = np.sum(action)
        if total < 1e-6:
            action = self.ratios / np.sum(self.ratios)
        else:
            action = action / total

        allocation = action * self.available_budget

        overspend = max(self.predicted_expense - self.available_budget, 0.0)

        reward = - (overspend / self.income)
        reward -= 0.01 * np.std(allocation)

        reward = float(np.clip(reward, -1.0, 1.0))

        done = True

        return self._get_state(), reward, done, {
            "allocation": allocation
        }
