from stable_baselines3 import PPO
from rl.budget_env import BudgetEnv
from config import CATEGORIES
import numpy as np

model = PPO.load("models/rl_budget_allocator")

def rl_allocate(income, savings, predicted_expense, ratios):
    env = BudgetEnv(
        income=income,
        savings=savings,
        predicted_expense=predicted_expense,
        ratios=ratios
    )

    state = env.reset()
    action, _ = model.predict(state)
    _, _, _, info = env.step(action)

    return info["allocation"]
