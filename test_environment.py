import pandas as pd

from environment.stock_trading_env import StockTradingEnv

# Small, predictable dataset for testing
test_data = pd.DataFrame(
    {
        "Close": [100.0, 110.0, 120.0],
        "Daily_Return": [0.0, 0.10, 0.09],
        "SMA_10": [100.0, 105.0, 110.0],
        "SMA_20": [100.0, 102.0, 105.0],
        "Volatility_10": [0.01, 0.02, 0.02],
        "RSI_14": [50.0, 60.0, 65.0],
    }
)

env = StockTradingEnv(
    test_data, initial_cash=1000.0, transaction_fee=0.001, trade_quantity=1
)

state = env.reset()

assert len(state) == 7
assert env.cash == 1000.0
assert env.holdings == 0

# Buy one share at the next day's closing price: $110
next_state, reward, done, info = env.step(1)

expected_cash = 1000.0 - 110.0 * 1.001
expected_value = expected_cash + 110.0

assert abs(info["cash"] - expected_cash) < 1e-6
assert info["holdings"] == 1
assert abs(info["portfolio_value"] - expected_value) < 1e-6
assert info["total_trades"] == 1

print("Environment validation passed!")
print("State size:", len(next_state))
print("Cash after buying:", round(info["cash"], 3))
print("Holdings:", info["holdings"])
print("Portfolio value:", round(info["portfolio_value"], 3))
print("Transaction fee: $0.110")


# Test Sell action in a fresh environment
sell_env = StockTradingEnv(
    test_data, initial_cash=1000.0, transaction_fee=0.001, trade_quantity=1
)

sell_env.reset()

# Buy at $110
sell_env.step(1)

# Sell at the next day's close: $120
next_state, reward, done, info = sell_env.step(2)

expected_cash_after_sell = 1000.0 - 110.0 * 1.001 + 120.0 * 0.999

assert info["holdings"] == 0
assert abs(info["cash"] - expected_cash_after_sell) < 1e-6
assert info["total_trades"] == 2

print("\nSell validation passed!")
print("Cash after selling:", round(info["cash"], 3))
print("Holdings:", info["holdings"])
print("Total trades:", info["total_trades"])

# Test Hold action
hold_env = StockTradingEnv(
    test_data, initial_cash=1000.0, transaction_fee=0.001, trade_quantity=1
)

hold_env.reset()
next_state, reward, done, info = hold_env.step(0)

assert info["cash"] == 1000.0
assert info["holdings"] == 0
assert info["total_trades"] == 0

print("\nHold validation passed!")
print("Cash:", info["cash"])
print("Holdings:", info["holdings"])
print("Total trades:", info["total_trades"])


# Test terminal-step behavior
terminal_env = StockTradingEnv(
    test_data, initial_cash=1000.0, transaction_fee=0.001, trade_quantity=1
)

terminal_env.reset()

# First step: day 0 -> day 1
_, _, done1, _ = terminal_env.step(0)
assert done1 is False

# Second step: day 1 -> day 2 (last day)
_, _, done2, info = terminal_env.step(0)
assert done2 is True

assert terminal_env.day == len(test_data) - 1

print("\nTerminal-step validation passed!")
print("Final day index:", terminal_env.day)
print("Done:", done2)
