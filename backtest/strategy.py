"""Long-only top-K trading strategy with transaction costs."""

import numpy as np
import pandas as pd


class LongOnlyTopK:
    """At each rebalance, hold the top-K stocks ranked by next-day predicted return.

    Round-trip transaction cost is applied on each rebalance proportional to the
    fraction of the portfolio that turned over (i.e., positions opened or closed).
    """

    def __init__(self, top_k: int = 5, cost_bps: float = 10.0):
        self.top_k = top_k
        self.cost_bps = cost_bps

    def run(
        self,
        predictions: pd.DataFrame,
        actual_returns: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        if not predictions.index.equals(actual_returns.index):
            common = predictions.index.intersection(actual_returns.index)
            predictions = predictions.loc[common]
            actual_returns = actual_returns.loc[common]

        tickers = predictions.columns.tolist()
        held = pd.Series(0.0, index=tickers)
        equity = 1.0

        gross_returns_series = []
        net_returns_series = []
        equity_series = []
        turnover_series = []

        for ts in predictions.index:
            ranks = predictions.loc[ts].rank(ascending=False, method="first")
            new_weights = pd.Series(0.0, index=tickers)
            top = ranks.nsmallest(self.top_k).index
            new_weights.loc[top] = 1.0 / self.top_k

            turnover = float((new_weights - held).abs().sum() / 2.0)
            cost = turnover * (self.cost_bps / 10_000.0)

            gross_ret = float((new_weights * actual_returns.loc[ts]).sum())
            net_ret = gross_ret - cost
            equity *= 1.0 + net_ret

            gross_returns_series.append(gross_ret)
            net_returns_series.append(net_ret)
            equity_series.append(equity)
            turnover_series.append(turnover)
            held = new_weights

        idx = predictions.index
        return {
            "gross_returns": pd.Series(gross_returns_series, index=idx),
            "net_returns": pd.Series(net_returns_series, index=idx),
            "equity": pd.Series(equity_series, index=idx),
            "turnover": pd.Series(turnover_series, index=idx),
        }


def equal_weighted_benchmark(actual_returns: pd.DataFrame, cost_bps: float = 1.0) -> dict[str, pd.Series]:
    """Equal-weighted buy-and-hold benchmark, with minimal rebalance cost."""
    n = actual_returns.shape[1]
    daily_ret = actual_returns.mean(axis=1)
    cost = (cost_bps / 10_000.0) / 252.0
    net_ret = daily_ret - cost
    equity = (1.0 + net_ret).cumprod()
    return {
        "gross_returns": daily_ret,
        "net_returns": net_ret,
        "equity": equity,
        "turnover": pd.Series(1.0 / n, index=actual_returns.index),
    }
