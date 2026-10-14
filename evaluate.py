import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from environment.stock_trading_env import StockTradingEnv
from agents.dqn_agent import DQNAgent

# Load processed historical data
df = pd.read_csv(
    "data/AAPL_processed_data.csv", index_col="Date", parse_dates=True
).sort_index()

feature_columns = ["Daily_Return", "SMA_10", "SMA_20", "Volatility_10", "RSI_14"]

# Same chronological split as training
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# Normalize using training data statistics only
means = train_df[feature_columns].mean()
stds = train_df[feature_columns].std().replace(0, 1).fillna(1)

test_df[feature_columns] = (test_df[feature_columns] - means) / stds

# Create test environment
initial_cash = 10000.0

env = StockTradingEnv(
    test_df, initial_cash=initial_cash, transaction_fee=0.001, trade_quantity=1
)

# Load trained DQN model
agent = DQNAgent(state_size=len(env.reset()), action_size=3)

agent.policy_net.load_state_dict(
    torch.load("outputs/dqn_model.pth", map_location="cpu", weights_only=True)
)

agent.policy_net.eval()


# Evaluate without exploration and record trading history
state = env.reset()
done = False

dates = [test_df.index[0]]
portfolio_values = [initial_cash]
actions = []
rewards = []
trade_log = []

while not done:
    current_day = env.day
    action = agent.act(state, explore=False)

    next_state, reward, done, info = env.step(action)

    action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}[action]

    execution_date = test_df.index[env.day]
    execution_price = float(test_df.iloc[env.day]["Close"])

    actions.append(action)
    rewards.append(reward)

    # Record each trading step
    trade_log.append(
        {
            "Date": execution_date,
            "Action": action_name,
            "Price": execution_price,
            "Cash": info["cash"],
            "Holdings": info["holdings"],
            "Portfolio_Value": info["portfolio_value"],
            "Reward": reward,
            "Executed_Trades": info["total_trades"],
        }
    )

    state = next_state
    portfolio_values.append(info["portfolio_value"])
    dates.append(execution_date)

# Save trading log
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

trade_log_df = pd.DataFrame(trade_log)

trade_log_df.to_csv(output_dir / "trading_log.csv", index=False)

print("\nTrading log saved successfully!")
print("File: outputs/trading_log.csv")
print("\nFirst five trading steps:")
print(trade_log_df.head())

print("\nExecuted trades:", info["total_trades"])


# Fair Buy-and-Hold benchmark:
# Whole shares and the same transaction fee as DQN

first_price = float(test_df["Close"].iloc[0])
last_price = float(test_df["Close"].iloc[-1])
fee = 0.001

shares = int(initial_cash // (first_price * (1 + fee)))
benchmark_cash = initial_cash - shares * first_price * (1 + fee)

benchmark_values = benchmark_cash + shares * test_df["Close"].reindex(dates).to_numpy()

final_value = portfolio_values[-1]
benchmark_value = float(benchmark_values[-1])

strategy_return = (final_value / initial_cash - 1) * 100
benchmark_return = (benchmark_value / initial_cash - 1) * 100

# Maximum drawdown
portfolio_series = pd.Series(portfolio_values, index=pd.to_datetime(dates))

daily_returns = portfolio_series.pct_change().dropna()

running_peak = portfolio_series.cummax()
drawdown = portfolio_series / running_peak - 1
max_drawdown = drawdown.min() * 100

# Annualized Sharpe ratio; assumes zero risk-free rate
if len(daily_returns) > 1 and daily_returns.std() > 0:
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
else:
    sharpe_ratio = np.nan

print("\n--- Updated DQN Evaluation ---")
print(f"Initial portfolio:       ${initial_cash:.2f}")
print(f"DQN final portfolio:     ${final_value:.2f}")
print(f"Buy-and-hold final:      ${benchmark_value:.2f}")
print(f"DQN total return:        {strategy_return:.2f}%")
print(f"Buy-and-hold return:     {benchmark_return:.2f}%")
print(f"Executed DQN trades:     {info['total_trades']}")
print(f"Maximum drawdown:        {max_drawdown:.2f}%")
print(f"Annualized Sharpe ratio: {sharpe_ratio:.3f}")

# Portfolio comparison
plt.figure(figsize=(12, 5))
plt.plot(dates, portfolio_values, label="DQN Strategy")
plt.plot(dates, benchmark_values, linestyle="--", label="Buy-and-Hold")
plt.title("DQN vs Buy-and-Hold — Test Period")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Drawdown chart
plt.figure(figsize=(12, 4))
plt.plot(drawdown.index, drawdown.values * 100)
plt.title("DQN Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


from pathlib import Path

# Create output folder
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Save portfolio performance history
results_df = pd.DataFrame(
    {"Date": pd.to_datetime(dates), "DQN_Portfolio_Value": portfolio_values}
)

results_df.to_csv(output_dir / "portfolio_results.csv", index=False)

# Save summary metrics
metrics_df = pd.DataFrame(
    [
        {
            "Initial_Capital": initial_cash,
            "DQN_Final_Value": final_value,
            "Buy_Hold_Final_Value": benchmark_value,
            "DQN_Return_Pct": strategy_return,
            "Buy_Hold_Return_Pct": benchmark_return,
            "Max_Drawdown_Pct": max_drawdown,
            "Sharpe_Ratio": sharpe_ratio,
            "Executed_Trades": info["total_trades"],
        }
    ]
)

metrics_df.to_csv(output_dir / "evaluation_metrics.csv", index=False)

# Save drawdown history
drawdown_df = pd.DataFrame(
    {"Date": drawdown.index, "Drawdown_Pct": drawdown.values * 100}
)

drawdown_df.to_csv(output_dir / "drawdown_results.csv", index=False)

print("\nEvaluation files saved successfully!")
print("outputs/portfolio_results.csv")
print("outputs/evaluation_metrics.csv")
print("outputs/drawdown_results.csv")
