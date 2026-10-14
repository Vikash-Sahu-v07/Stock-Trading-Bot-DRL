import pandas as pd
from environment.stock_trading_env import StockTradingEnv


def make_env():
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

    return StockTradingEnv(
        test_data,
        initial_cash=1000.0,
        transaction_fee=0.001,
        trade_quantity=1,
    )


def test_buy_action():
    env = make_env()
    state = env.reset()

    assert len(state) == 7
    assert env.cash == 1000.0
    assert env.holdings == 0

    _, _, _, info = env.step(1)

    expected_cash = 1000.0 - 110.0 * 1.001
    expected_value = expected_cash + 110.0

    assert abs(info["cash"] - expected_cash) < 1e-6
    assert info["holdings"] == 1
    assert abs(info["portfolio_value"] - expected_value) < 1e-6
    assert info["total_trades"] == 1


def test_sell_action():
    env = make_env()
    env.reset()

    env.step(1)
    _, _, _, info = env.step(2)

    expected_cash = 1000.0 - 110.0 * 1.001 + 120.0 * 0.999

    assert info["holdings"] == 0
    assert abs(info["cash"] - expected_cash) < 1e-6
    assert info["total_trades"] == 2


def test_hold_action():
    env = make_env()
    env.reset()

    _, _, _, info = env.step(0)

    assert info["cash"] == 1000.0
    assert info["holdings"] == 0
    assert info["total_trades"] == 0


def test_terminal_step():
    env = make_env()
    env.reset()

    _, _, done1, _ = env.step(0)
    assert done1 is False

    _, _, done2, _ = env.step(0)
    assert done2 is True

    assert env.day == 2
