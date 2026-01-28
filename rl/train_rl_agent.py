import os
from stable_baselines3 import PPO
from rl.budget_env import BudgetEnv
from config import CATEGORIES

os.makedirs("models", exist_ok=True)

# Fixed category ratios (same length always)
ratios = [1 / len(CATEGORIES)] * len(CATEGORIES)

env = BudgetEnv(
    income=20000,
    savings=5000,
    predicted_expense=12000,
    ratios=ratios
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003
)

model.learn(total_timesteps=10000)

model.save("models/rl_budget_allocator")

print("✅ RL agent trained and saved")
