# AI-Based Future Expense Prediction and Optimal Budget Allocation using LSTM and Reinforcement Learning 💰

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
git clone https://github.com/Neeraj0608/lstm-rl-budget-planner.git
cd lstm-rl-budget-planner
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ (Optional) Train models
```
python -m lstm.train_lstm_model
python -m rl.train_rl_agent
```
### 4️⃣ Run the application
```
streamlit run app.py
```
---

## 🏗 Detailed Workflow
- User uploads historical expense data
- Data is cleaned and aggregated
- LSTM model predicts future daily and monthly expenses
- Reinforcement Learning (RL) agent receives:
  - Category-wise spending ratios
  - Predicted future expenses
  - Available budget *(income − savings)*
- RL agent optimally allocates the budget
-  across expense categories
- Results are visualized on the dashboard

### 💫 Results
<img width="1818" height="796" alt="image" src="https://github.com/user-attachments/assets/71949cf9-31b4-44de-a217-16358757236c" />
<img width="1810" height="707" alt="image" src="https://github.com/user-attachments/assets/ffbd9ef0-167b-4014-b2fd-47bf997c0e77" />


