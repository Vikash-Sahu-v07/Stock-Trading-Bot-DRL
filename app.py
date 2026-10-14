import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="DQN Stock Trading Dashboard", page_icon="📈", layout="wide"
)

st.title("Stock Trading Bot — DQN Dashboard")
st.caption("Educational simulation | Historical test-period results")

output_dir = Path("outputs")

metrics_file = output_dir / "evaluation_metrics.csv"
portfolio_file = output_dir / "portfolio_results.csv"
trades_file = output_dir / "trading_log.csv"
drawdown_file = output_dir / "drawdown_results.csv"

if not metrics_file.exists() or not portfolio_file.exists():
    st.error("Evaluation files missing. Run python evaluate.py first.")
    st.stop()

metrics = pd.read_csv(metrics_file).iloc[0]
portfolio = pd.read_csv(portfolio_file, parse_dates=["Date"])

# Top-level KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("DQN Return", f"{metrics['DQN_Return_Pct']:.2f}%")

col2.metric("Buy-and-Hold Return", f"{metrics['Buy_Hold_Return_Pct']:.2f}%")

col3.metric("Maximum Drawdown", f"{metrics['Max_Drawdown_Pct']:.2f}%")

col4.metric("Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.3f}")

st.divider()

# Portfolio chart
st.subheader("Portfolio Performance")

portfolio = portfolio.sort_values("Date").set_index("Date")
st.line_chart(portfolio["DQN_Portfolio_Value"])

st.caption(
    "This chart shows the DQN portfolio history. "
    "Benchmark comparison is displayed below."
)

# Benchmark comparison
st.subheader("DQN vs Buy-and-Hold")

comparison = pd.DataFrame(
    {
        "Strategy": ["DQN", "Buy-and-Hold"],
        "Final Portfolio Value": [
            metrics["DQN_Final_Value"],
            metrics["Buy_Hold_Final_Value"],
        ],
    }
).set_index("Strategy")

st.bar_chart(comparison)

# Drawdown chart
if drawdown_file.exists():
    st.subheader("Portfolio Drawdown")
    drawdown = pd.read_csv(drawdown_file, parse_dates=["Date"]).set_index("Date")

    st.line_chart(drawdown["Drawdown_Pct"])


# Stock price and trading signals
if trades_file.exists():
    st.subheader("Stock Price & Trading Signals")

    price_data = pd.read_csv(
        "data/AAPL_processed_data.csv", index_col="Date", parse_dates=True
    ).sort_index()

    trades = pd.read_csv(trades_file, parse_dates=["Date"])

    # Keep only dates present in the test-period log
    price_data = price_data.loc[price_data.index.isin(trades["Date"])]

    buy_dates = trades.loc[trades["Action"] == "BUY", "Date"]

    sell_dates = trades.loc[trades["Action"] == "SELL", "Date"]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(price_data.index, price_data["Close"], label="AAPL Closing Price")

    buy_points = price_data.loc[price_data.index.isin(buy_dates)]

    sell_points = price_data.loc[price_data.index.isin(sell_dates)]

    ax.scatter(buy_points.index, buy_points["Close"], marker="^", label="Buy", s=70)

    ax.scatter(sell_points.index, sell_points["Close"], marker="v", label="Sell", s=70)

    ax.set_title("DQN Trading Decisions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Closing Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# Trading history
if trades_file.exists():
    st.subheader("Trading History")

    trades = pd.read_csv(trades_file, parse_dates=["Date"])

    actions = ["All"] + sorted(trades["Action"].unique().tolist())
    selected_action = st.selectbox("Filter by action", actions)

    filtered = trades.copy()

    if selected_action != "All":
        filtered = filtered[filtered["Action"] == selected_action]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "Download trading log CSV",
        data=trades.to_csv(index=False).encode("utf-8"),
        file_name="trading_log.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Research prototype only. Historical performance does not "
    "guarantee future returns. Not financial advice."
)
