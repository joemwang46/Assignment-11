import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Portfolio:
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}
        self.last_prices = {}
        self.portfolio_value = initial_cash

        self.dates = []
        self.equity_curve = []
        self.pnl_curve = []
        self.position_curve = []

    def update_prices(self, ticker: str, price: float):
        self.last_prices[ticker] = price

    def _revalue_portfolio(self):
        holdings_value = sum(qty * self.last_prices.get(tkr, 0)
                             for tkr, qty in self.holdings.items())
        self.portfolio_value = self.cash + holdings_value
        return holdings_value

    def buy_ticker(self, ticker: str, price: float):
        if self.cash >= price:
            self.holdings[ticker] = self.holdings.get(ticker, 0) + 1
            self.cash -= price
        else:
            pass

        self._revalue_portfolio()

    def log_day(self, day):
        holdings_value = sum(qty * self.last_prices.get(tkr, 0)
                             for tkr, qty in self.holdings.items())
        total_exposure = holdings_value

        equity = self.portfolio_value
        pnl = equity - self.initial_cash

        self.dates.append(day)
        self.equity_curve.append(equity)
        self.pnl_curve.append(pnl)
        self.position_curve.append(total_exposure)


def run_model_backtest(model_storage: dict, data: pd.DataFrame, tickers: list, features: list, initial_cash: float = 100000):
    portfolio = Portfolio(initial_cash)

    for day, group in data.groupby(level=0):

        for _, row in group.iterrows():
            portfolio.update_prices(row["ticker"], row["close"])

        for idx, row in group.iterrows():
            model = model_storage.get(row["ticker"])
            if model is None:
                continue

            row_df = group.loc[[idx]]
            signal = model.predict(row_df[features])[0]

            if signal == 1:
                portfolio.buy_ticker(row["ticker"], row["close"])

        portfolio.log_day(day)

    return portfolio


def run_benchmark_backtest(data: pd.DataFrame, tickers: list, initial_cash: float = 100000):
    portfolio = Portfolio(initial_cash)

    for day, group in data.groupby(level=0):

        for _, row in group.iterrows():
            portfolio.update_prices(row["ticker"], row["close"])

        for _, row in group.iterrows():
            portfolio.buy_ticker(row["ticker"], row["close"])

        portfolio.log_day(day)

    return portfolio

def plot_backtest(portfolio: Portfolio, title="Backtest Results"):
    plt.figure(figsize=(12, 8))

    plt.subplot(3,1,1)
    plt.plot(portfolio.dates, portfolio.equity_curve)
    plt.title(f"{title} — Equity Curve")
    plt.ylabel("Equity")

    plt.subplot(3,1,2)
    plt.plot(portfolio.dates, portfolio.pnl_curve)
    plt.title("PnL Over Time")
    plt.ylabel("PnL")

    plt.subplot(3,1,3)
    plt.plot(portfolio.dates, portfolio.position_curve)
    plt.title("Position Exposure Over Time")
    plt.ylabel("Exposure ($)")
    plt.xlabel("Date")

    plt.tight_layout()
    plt.show()
