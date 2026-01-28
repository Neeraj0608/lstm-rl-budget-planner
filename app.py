import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# IMPORTS (LSTM + RL + CONFIG)
# --------------------------------------------------
from lstm.lstm_predictor import lstm_predict
from rl.rl_allocator import rl_allocate
from config import CATEGORIES

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI Budget Optimizer", layout="wide")
st.title("💰 AI Budget Optimizer (LSTM + Reinforcement Learning)")

# --------------------------------------------------
# INPUT DATA
# --------------------------------------------------
st.header("📂 Upload Expense Data")

file = st.file_uploader(
    "Upload expense file (CSV or Excel with Date, Category, Amount)",
    type=["csv", "xlsx"]
)

if not file:
    st.stop()

df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Category"] = df["Category"].astype(str).str.strip()
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

df = df.dropna()
df = df[df["Amount"] > 0]

if df.empty:
    st.error("No valid expense data found.")
    st.stop()

st.subheader("📄 Cleaned Expense Data")
st.dataframe(df)

# --------------------------------------------------
# AGGREGATIONS
# --------------------------------------------------
daily = df.groupby("Date")["Amount"].sum()
category_sum = df.groupby("Category")["Amount"].sum()

# --------------------------------------------------
# LSTM PREDICTION
# --------------------------------------------------
st.header("🔮 Expense Prediction (LSTM)")

predicted = None
predicted_month = 0

if len(daily.values) >= 3:
    predicted = lstm_predict(daily.values, days=7)
    predicted_month = int(predicted.sum() * 4)

    fig = px.line(
        y=predicted.flatten(),
        markers=True,
        labels={"y": "Amount"},
        title="Predicted Daily Expenses (Next 7 Days)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "Estimated Monthly Expense (Extrapolated)",
        f"₹{predicted_month}"
    )
else:
    st.warning("Not enough data for LSTM prediction (need at least 3 days).")

# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------
st.header("💼 Financial Inputs")

income = st.number_input("Monthly Income (₹)", value=20000)
savings = st.number_input("Savings Goal (₹)", value=5000)

available_budget = max(income - savings, 0)

st.metric("Available Budget After Savings", f"₹{available_budget}")

# --------------------------------------------------
# RL BUDGET ALLOCATION
# --------------------------------------------------
st.header("🤖 RL-Based Budget Allocation")

# FIXED CATEGORY ORDER → FIXED STATE SIZE
ratios = np.array(
    [category_sum.get(cat, 0) for cat in CATEGORIES],
    dtype=np.float32
)

# Normalize safely
if ratios.sum() == 0:
    ratios = np.ones(len(CATEGORIES), dtype=np.float32) / len(CATEGORIES)
else:
    ratios = ratios / ratios.sum()

# Call RL agent
allocation = rl_allocate(
    income=income,
    savings=savings,
    predicted_expense=predicted_month,
    ratios=ratios
)

alloc_df = pd.DataFrame({
    "Category": CATEGORIES,
    "Allocated Budget (₹)": allocation.astype(int),
    "Historical Spend (₹)": [category_sum.get(cat, 0) for cat in CATEGORIES]
})

alloc_df["Status"] = np.where(
    alloc_df["Allocated Budget (₹)"] < alloc_df["Historical Spend (₹)"],
    "Needs Reduction",
    "Within Limit"
)

st.subheader("📊 Budget Allocation Result")
st.dataframe(alloc_df)

fig = px.bar(
    alloc_df,
    x="Category",
    y="Allocated Budget (₹)",
    color="Status",
    title="RL-Optimized Budget Allocation"
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TREND ANALYSIS
# --------------------------------------------------
st.header("📈 Expense Trend")

trend_fig = px.line(
    daily.reset_index(),
    x="Date",
    y="Amount",
    markers=True,
    title="Daily Expense Trend"
)
st.plotly_chart(trend_fig, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(
    "Final Year Project | LSTM for Forecasting + PPO Reinforcement Learning for Budget Allocation"
)
