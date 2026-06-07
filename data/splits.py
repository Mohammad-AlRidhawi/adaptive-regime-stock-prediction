"""Strict chronological train/validation/test splits."""

import pandas as pd


def chronological_split(
    data: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DatetimeIndex-keyed dataframe into train/val/test at the given dates."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = data.loc[data.index <= train_end_ts]
    val = data.loc[(data.index > train_end_ts) & (data.index <= val_end_ts)]
    test = data.loc[data.index > val_end_ts]

    return train, val, test


def select_stable_vix_days(
    vix: pd.Series,
    percentile: float = 75.0,
    train_end: str = "2010-12-31",
) -> pd.DatetimeIndex:
    """Return the trading dates with training-period VIX below the given percentile."""
    train_vix = vix.loc[vix.index <= pd.Timestamp(train_end)]
    threshold = train_vix.quantile(percentile / 100.0)
    return vix.loc[vix <= threshold].index
