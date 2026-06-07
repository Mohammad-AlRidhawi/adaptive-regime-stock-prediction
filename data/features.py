"""Feature engineering: technical indicators + sentiment integration.

Produces:
  - prediction features x_{i,t} in R^17 (eq. 2 of the manuscript)
  - router features x_{i,t}^router in R^6 (eq. 3 of the manuscript)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


PREDICTION_FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "sma_5", "sma_10", "sma_20",
    "ema_5", "ema_10", "ema_20",
    "rsi_14", "macd",
    "return_1d", "log_return_1d", "vol_20d",
    "sentiment",
]

ROUTER_FEATURE_COLUMNS = [
    "vol_5d", "vol_20d", "delta_vix", "delta_corr", "abs_sentiment", "post_velocity",
]


def build_prediction_features(
    ohlcv: pd.DataFrame,
    sentiment: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute the 17-dimensional prediction features for a single stock."""
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]

    df["sma_5"] = ta.sma(df["close"], length=5)
    df["sma_10"] = ta.sma(df["close"], length=10)
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["ema_5"] = ta.ema(df["close"], length=5)
    df["ema_10"] = ta.ema(df["close"], length=10)
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]

    df["return_1d"] = df["close"].pct_change()
    df["log_return_1d"] = np.log(df["close"]).diff()
    df["vol_20d"] = df["return_1d"].rolling(20).std()

    if sentiment is not None:
        df["sentiment"] = sentiment.reindex(df.index).fillna(0.0)
    else:
        df["sentiment"] = 0.0

    return df[PREDICTION_FEATURE_COLUMNS]


def build_router_features(
    ohlcv: pd.DataFrame,
    vix: pd.Series,
    cross_correlation: pd.Series,
    sentiment: pd.Series | None = None,
    post_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute the 6-dimensional router features for a single stock."""
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    returns = df["close"].pct_change()

    out = pd.DataFrame(index=df.index)
    out["vol_5d"] = returns.rolling(5).std()
    out["vol_20d"] = returns.rolling(20).std()
    out["delta_vix"] = vix.reindex(df.index).pct_change()
    out["delta_corr"] = cross_correlation.reindex(df.index).diff()

    if sentiment is not None:
        out["abs_sentiment"] = sentiment.reindex(df.index).abs().fillna(0.0)
    else:
        out["abs_sentiment"] = 0.0

    if post_counts is not None:
        out["post_velocity"] = post_counts.reindex(df.index).fillna(0.0)
    else:
        out["post_velocity"] = 0.0

    return out[ROUTER_FEATURE_COLUMNS]


def build_event_context(
    reconstruction_errors: pd.DataFrame,
    sentiment: pd.Series,
    sentiment_train_std: float,
    earnings_dates: pd.DatetimeIndex | None = None,
    sector_surprise: pd.Series | None = None,
) -> pd.DataFrame:
    """Construct the 12-dimensional event context vector c_t per timestep."""
    idx = reconstruction_errors.index

    # Sentiment-spike (binary flag + scaled magnitude)
    spike_flag = (sentiment.abs() > 2 * sentiment_train_std).astype(float)
    spike_magnitude = (sentiment.abs() / (sentiment_train_std + 1e-9)).clip(0, 5) / 5.0

    # Event characterization (4 dims)
    if earnings_dates is not None and len(earnings_dates) > 0:
        next_earnings = pd.Series(index=idx, dtype=float)
        for ts in idx:
            future = earnings_dates[earnings_dates >= ts]
            if len(future) > 0:
                next_earnings[ts] = (future[0] - ts).days
            else:
                next_earnings[ts] = 90.0
        days_to_earnings = next_earnings.clip(0, 90) / 90.0
        in_window = (next_earnings <= 7).astype(float)
    else:
        days_to_earnings = pd.Series(1.0, index=idx)
        in_window = pd.Series(0.0, index=idx)

    surprise_history = sector_surprise.reindex(idx).fillna(0.0) if sector_surprise is not None else pd.Series(0.0, index=idx)
    sector_avg_surprise = surprise_history.rolling(20).mean().fillna(0.0)

    # Cross-asset stress (2 dims)
    cross_mean = reconstruction_errors.mean(axis=1)
    cross_std = reconstruction_errors.std(axis=1)

    context = pd.DataFrame(
        {
            "spike_flag": spike_flag.reindex(idx).fillna(0.0),
            "spike_magnitude": spike_magnitude.reindex(idx).fillna(0.0),
            "days_to_earnings": days_to_earnings,
            "earnings_in_window": in_window,
            "surprise_history": surprise_history,
            "sector_avg_surprise": sector_avg_surprise,
            "cross_mean": cross_mean,
            "cross_std": cross_std,
        }
    )

    # The regime embedding (4 dims) is produced by the trainable embedding layer
    # inside NodeTransformer at forward time, so we expose only the 8 continuous
    # context dimensions here; the embedding layer concatenates the remaining 4.
    return context
