"""Expanding-window z-score normalization (eq. 1) to prevent look-ahead bias."""

import numpy as np
import pandas as pd


class ExpandingZScore:
    """Per-feature expanding-window z-score with frozen statistics for val/test.

    During training, statistics at time t use only data from the first training
    observation through time t. During validation and test, statistics are fixed
    at the full training-period values (mu_{1:T_train}, sigma_{1:T_train}).
    """

    def __init__(self):
        self.train_mean: pd.Series | None = None
        self.train_std: pd.Series | None = None

    def fit(self, train_data: pd.DataFrame) -> "ExpandingZScore":
        self.train_mean = train_data.mean(skipna=True)
        self.train_std = train_data.std(skipna=True).replace(0.0, 1.0)
        return self

    def transform_expanding(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply expanding-window z-score (training mode)."""
        mean = data.expanding(min_periods=1).mean()
        std = data.expanding(min_periods=1).std().replace(0.0, 1.0).fillna(1.0)
        return (data - mean) / std

    def transform_frozen(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen training-period z-score (validation/test mode)."""
        if self.train_mean is None or self.train_std is None:
            raise RuntimeError("ExpandingZScore.fit() must be called before transform_frozen.")
        return (data - self.train_mean) / self.train_std

    def transform(self, data: pd.DataFrame, frozen: bool = False) -> pd.DataFrame:
        return self.transform_frozen(data) if frozen else self.transform_expanding(data)
