import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Project data folder
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# 2. Download historical stock data
ticker = "AAPL"

print("Downloading historical stock data...")

df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)

# 3. Check whether data was downloaded
if df.empty:
    raise RuntimeError(
        "No data downloaded. Check your internet connection " "and try again."
    )

# Handle column format returned by different yfinance versions
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 4. Clean and inspect data
df = df.sort_index()
df = df[~df.index.duplicated(keep="last")]
df = df.dropna(subset=["Close"])

# 5. Save data locally
file_path = DATA_DIR / "AAPL_historical_data.csv"
df.to_csv(file_path, index_label="Date")

# 6. Display basic information
print("\nData downloaded successfully!")
print("Stock:", ticker)
print("Number of rows:", len(df))
print("First date:", df.index.min().date())
print("Last date:", df.index.max().date())
print("\nFirst 5 rows:")
print(df.head())


# Step 8: Feature Engineering

# Daily percentage return
df["Daily_Return"] = df["Close"].pct_change()

# Moving averages
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()

# Daily price volatility
df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()

# RSI (Relative Strength Index)
delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

relative_strength = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + relative_strength))

# Remove rows where indicators are unavailable
df = df.dropna().copy()

# Save processed dataset
processed_path = DATA_DIR / "AAPL_processed_data.csv"
df.to_csv(processed_path, index_label="Date")

print("\nFeature engineering completed!")
print("\nProcessed dataset:")
print(
    df[["Close", "Daily_Return", "SMA_10", "SMA_20", "Volatility_10", "RSI_14"]].tail()
)

print("\nProcessed dataset saved to:", processed_path)


# 7. Plot closing price
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Close"], label="AAPL Closing Price")
plt.title("Apple Stock - Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Adjusted Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nDataset saved to: {file_path}")


from environment.stock_trading_env import StockTradingEnv

# Load processed data
processed_df = pd.read_csv(
    "data/AAPL_processed_data.csv", index_col="Date", parse_dates=True
)

# Create the environment
env = StockTradingEnv(
    processed_df, initial_cash=10000, transaction_fee=0.001, trade_quantity=1
)

state = env.reset()

print("\nTrading environment created successfully!")
print("State shape:", state.shape)
print("Initial state:", state)

# Test one Buy action
next_state, reward, done, info = env.step(1)

print("\nAfter one Buy action:")
print("Reward:", round(reward, 2))
print("Cash:", round(info["cash"], 2))
print("Holdings:", info["holdings"])
print("Portfolio value:", round(info["portfolio_value"], 2))


from agents.dqn_agent import DQNAgent

# Create the DQN agent using the environment state size
agent = DQNAgent(state_size=len(state), action_size=3)

# Test action selection
action = agent.act(state)

print("\nDQN agent created successfully!")
print("State size:", len(state))
print("Available actions: Hold (0), Buy (1), Sell (2)")
print("Selected action:", action)
print("Initial epsilon:", agent.epsilon)
print("Replay memory size:", len(agent.memory))
