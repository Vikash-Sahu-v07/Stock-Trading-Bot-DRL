import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from environment.stock_trading_env import StockTradingEnv
from agents.dqn_agent import DQNAgent

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load processed data
df = pd.read_csv(
    "data/AAPL_processed_data.csv", index_col="Date", parse_dates=True
).sort_index()

feature_columns = ["Daily_Return", "SMA_10", "SMA_20", "Volatility_10", "RSI_14"]

# Chronological split: first 80% for training
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

if len(train_df) < 2 or len(test_df) < 2:
    raise ValueError("Not enough data for train/test split.")

# Fit normalization on training data only
means = train_df[feature_columns].mean()
stds = train_df[feature_columns].std().replace(0, 1).fillna(1)

train_df[feature_columns] = (train_df[feature_columns] - means) / stds

test_df[feature_columns] = (test_df[feature_columns] - means) / stds

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print(
    "Training period:", train_df.index.min().date(), "to", train_df.index.max().date()
)
print("Testing period:", test_df.index.min().date(), "to", test_df.index.max().date())

# Create training environment and agent
env = StockTradingEnv(
    train_df, initial_cash=10000, transaction_fee=0.001, trade_quantity=1
)

state_size = len(env.reset())

agent = DQNAgent(
    state_size=state_size, action_size=3, learning_rate=0.001, gamma=0.95, batch_size=64
)

episodes = 50
target_update_frequency = 5

print("\nStarting DQN training...")

for episode in range(1, episodes + 1):
    state = env.reset()
    total_reward = 0.0
    losses = []
    done = False

    while not done:
        action = agent.act(state, explore=True)

        next_state, reward, done, info = env.step(action)

        agent.remember(state, action, reward, next_state, done)

        loss = agent.replay()

        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward

    agent.decay_epsilon()

    if episode % target_update_frequency == 0:
        agent.update_target_network()

    avg_loss = np.mean(losses) if losses else 0.0

    print(
        f"Episode {episode:02d}/{episodes} | "
        f"Reward: {total_reward:.2f} | "
        f"Trades: {info['total_trades']} | "
        f"Epsilon: {agent.epsilon:.3f} | "
        f"Loss: {avg_loss:.4f}"
    )

# Save trained model
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

torch.save(agent.policy_net.state_dict(), output_dir / "dqn_model.pth")

print("\nTraining completed!")
print("Model saved to outputs/dqn_model.pth")
