"""Sentiment loader: pre-computed BERT scores per stock per day."""

from pathlib import Path

import pandas as pd


class SentimentLoader:
    """Loads the CSS daily sentiment scores produced by the fine-tuned BERT classifier.

    Sentiment is computed offline by the BERT pipeline released alongside the
    prior NodeFormer-BERT work (alridhawi2025nodeformer). For each stock and
    trading day, the score is in [-1, +1]; days with no posts default to 0.
    """

    def __init__(self, parquet_path: str | Path):
        self.path = Path(parquet_path)
        self._frame: pd.DataFrame | None = None

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = pd.read_parquet(self.path)
            self._frame.index = pd.to_datetime(self._frame.index)
            self._frame = self._frame.sort_index()
        return self._frame

    def get_series(self, ticker: str) -> pd.Series:
        df = self.frame
        if ticker not in df.columns:
            return pd.Series(dtype=float)
        return df[ticker]

    def aggregate_to_trading_days(self, ticker: str, trading_days: pd.DatetimeIndex) -> pd.Series:
        """Attribute weekend/holiday sentiment to the next trading day."""
        s = self.get_series(ticker)
        if s.empty:
            return pd.Series(0.0, index=trading_days)
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0.0)
        aligned = pd.Series(index=trading_days, dtype=float)
        for i, ts in enumerate(trading_days):
            prev_ts = trading_days[i - 1] if i > 0 else None
            if prev_ts is None:
                window = s.loc[:ts]
            else:
                window = s.loc[prev_ts:ts]
            aligned[ts] = window.mean() if not window.empty else 0.0
        return aligned.fillna(0.0)
