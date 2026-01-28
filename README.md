# AI Budget Optimizer 💰

An intelligent personal finance system that predicts future expenses using **LSTM** and allocates budgets optimally using **Reinforcement Learning (PPO)**.

---

## 🚀 Features
- Expense forecasting using LSTM (time-series prediction)
- Intelligent budget allocation using Reinforcement Learning
- Supports CSV expense uploads
- Explainable budget recommendations
- Streamlit-based interactive dashboard

---

## 🧠 System Architecture
1. LSTM predicts future expenses based on historical data  
2. Reinforcement Learning agent allocates budget optimally  
3. User dashboard visualizes predictions and allocations  

---

## ▶️ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Neeraj0608/AI-Budget-Optimizer.git
cd AI-Budget-Optimizer

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt

### 3️⃣ (Optional) Train models
python -m lstm.train_lstm_model
python -m rl.train_rl_agent

### 4️⃣ Run the application

streamlit run app.py

---
🏗 Detailed Working Flow

User uploads historical expense data
Data is cleaned and aggregated
LSTM predicts future daily and monthly expenses
RL agent receives:
  Category-wise spending ratios
  Predicted future expense
  Available budget (income − savings)
  RL agent allocates budget optimally across categories

Results are visualized on the dashboard