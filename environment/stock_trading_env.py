import numpy as np


class StockTradingEnv:
    """
    Simple historical stock trading environment.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    """

    def __init__(
        self, data, initial_cash=10000.0, transaction_fee=0.001, trade_quantity=1
    ):
        self.data = data.reset_index(drop=True).copy()

        self.initial_cash = float(initial_cash)
        self.transaction_fee = float(transaction_fee)
        self.trade_quantity = int(trade_quantity)

        self.feature_columns = [
            "Daily_Return",
            "SMA_10",
            "SMA_20",
            "Volatility_10",
            "RSI_14",
        ]

        required = ["Close"] + self.feature_columns
        missing = set(required) - set(self.data.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if len(self.data) < 2:
            raise ValueError("At least 2 rows of data are required.")

        if self.data[required].isna().any().any():
            raise ValueError("Data contains missing feature values.")

        self.reset()

    def reset(self):
        self.day = 0
        self.cash = self.initial_cash
        self.holdings = 0
        self.total_trades = 0

        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.day]

        market_features = row[self.feature_columns].to_numpy(dtype=np.float32)

        # Portfolio information, scaled for numerical stability
        price = float(row["Close"])

        portfolio_features = np.array(
            [self.cash / self.initial_cash, self.holdings * price / self.initial_cash],
            dtype=np.float32,
        )

        return np.concatenate([market_features, portfolio_features])

    def _portfolio_value(self, price):
        return self.cash + self.holdings * price

    def step(self, action):
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0, 1, or 2.")

        current_price = float(self.data.iloc[self.day]["Close"])
        old_value = self._portfolio_value(current_price)

        # Move to the next trading day.
        # The action is executed at the next day's close.
        self.day += 1
        execution_price = float(self.data.iloc[self.day]["Close"])

        # Action 1: Buy
        if action == 1:
            cost = execution_price * self.trade_quantity
            total_cost = cost * (1 + self.transaction_fee)

            if self.cash >= total_cost:
                self.cash -= total_cost
                self.holdings += self.trade_quantity
                self.total_trades += 1

        # Action 2: Sell
        elif action == 2 and self.holdings > 0:
            quantity = min(self.trade_quantity, self.holdings)
            proceeds = execution_price * quantity
            net_proceeds = proceeds * (1 - self.transaction_fee)

            self.cash += net_proceeds
            self.holdings -= quantity
            self.total_trades += 1

        new_value = self._portfolio_value(execution_price)

        # Reward includes the price movement and transaction costs.
        reward = new_value - old_value
        done = self.day >= len(self.data) - 1

        info = {
            "cash": self.cash,
            "holdings": self.holdings,
            "portfolio_value": new_value,
            "total_trades": self.total_trades,
        }

        return self._get_state(), reward, done, info
