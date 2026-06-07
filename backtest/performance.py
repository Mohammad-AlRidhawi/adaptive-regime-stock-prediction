"""Risk-adjusted performance metrics: Sharpe, Sortino, Calmar, max drawdown, win rate."""

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_returns: pd.Series) -> float:
    cumulative = float((1.0 + daily_returns).prod())
    years = max(len(daily_returns) / TRADING_DAYS_PER_YEAR, 1e-9)
    return cumulative ** (1.0 / years) - 1.0


def annualized_volatility(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    ann_ret = annualized_return(daily_returns) - risk_free_rate
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol < 1e-9:
        return 0.0
    return float(ann_ret / ann_vol)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    downside = daily_returns[daily_returns < risk_free_rate / TRADING_DAYS_PER_YEAR]
    downside_std = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 1 else 1e-9
    ann_ret = annualized_return(daily_returns) - risk_free_rate
    return float(ann_ret / max(downside_std, 1e-9))


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def calmar_ratio(daily_returns: pd.Series, equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd < 1e-9:
        return 0.0
    return float(annualized_return(daily_returns) / mdd)


def win_rate(daily_returns: pd.Series) -> float:
    return float((daily_returns > 0).mean())


def compute_backtest_metrics(
    backtest_output: dict[str, pd.Series],
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    gross = backtest_output["gross_returns"]
    net = backtest_output["net_returns"]
    equity = backtest_output["equity"]
    return {
        "annualized_return_gross": annualized_return(gross),
        "annualized_return_net": annualized_return(net),
        "annualized_volatility": annualized_volatility(net),
        "sharpe_net": sharpe_ratio(net, risk_free_rate),
        "sortino_net": sortino_ratio(net, risk_free_rate),
        "max_drawdown": max_drawdown(equity),
        "calmar_net": calmar_ratio(net, equity),
        "win_rate": win_rate(net),
    }
