import pandas as pd
import numpy as np
from backtest import Portfolio, run_model_backtest, run_benchmark_backtest

class DummyModel:
    def predict(self, X):
        return np.ones(len(X))

def test_portfolio_buy_and_valuation():
    p = Portfolio(initial_cash=100)
    p.update_prices("AAPL", 10)

    p.buy_ticker("AAPL", 10)
    assert p.cash == 90
    assert p.holdings["AAPL"] == 1

    value = p._revalue_portfolio()
    assert value == 10
    assert p.portfolio_value == 100


def test_portfolio_logging():
    p = Portfolio(initial_cash=100)
    p.update_prices("AAPL", 10)
    p.buy_ticker("AAPL", 10)

    p.log_day("2025-01-01")

    assert len(p.dates) == 1
    assert len(p.equity_curve) == 1
    assert len(p.pnl_curve) == 1
    assert len(p.position_curve) == 1


def test_run_model_backtest():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3, freq="D"),
        "ticker": ["AAPL", "AAPL", "AAPL"],
        "close": [10, 11, 12],
        "f1": [1, 1, 1]
    }).set_index("date")

    model_storage = {"AAPL": DummyModel()}
    p = run_model_backtest(model_storage, df, ["AAPL"], ["f1"])

    assert len(p.equity_curve) == 3
    assert len(p.pnl_curve) == 3


def test_run_benchmark_backtest():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3, freq="D"),
        "ticker": ["AAPL", "AAPL", "AAPL"],
        "close": [10, 11, 12],
    }).set_index("date")

    p = run_benchmark_backtest(df, ["AAPL"])

    assert len(p.equity_curve) == 3
    assert p.holdings["AAPL"] >= 1
