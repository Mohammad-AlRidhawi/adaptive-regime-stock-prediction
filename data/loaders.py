"""Dataset and DataLoader builders for OHLCV + sentiment + per-stock targets."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .features import build_prediction_features
from .normalization import ExpandingZScore
from .stocks import STOCK_UNIVERSE, ticker_to_id


class OHLCVDataset(Dataset):
    """Per-stock 252-day rolling windows with H-day-ahead closing-price targets."""

    def __init__(
        self,
        features_by_stock: dict[str, pd.DataFrame],
        sequence_length: int = 252,
        horizons: tuple[int, ...] = (1, 5, 20),
    ):
        self.features_by_stock = features_by_stock
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.samples: list[tuple[str, int]] = []

        for ticker, df in features_by_stock.items():
            max_h = max(horizons)
            for end_idx in range(sequence_length, len(df) - max_h):
                if df.iloc[end_idx - sequence_length:end_idx].isna().any().any():
                    continue
                self.samples.append((ticker, end_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ticker, end_idx = self.samples[idx]
        df = self.features_by_stock[ticker]
        window = df.iloc[end_idx - self.sequence_length:end_idx]
        features = torch.from_numpy(window.values.astype(np.float32))
        targets = {}
        for h in self.horizons:
            future_close = df["close"].iloc[end_idx + h - 1]
            targets[f"y_{h}"] = torch.tensor(future_close, dtype=torch.float32)
        return {
            "features": features,
            "stock_id": torch.tensor(ticker_to_id(ticker), dtype=torch.long),
            **targets,
        }


def load_raw_ohlcv(ohlcv_dir: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load per-ticker OHLCV CSVs from disk."""
    out = {}
    for t in tickers:
        path = Path(ohlcv_dir) / f"{t}.csv"
        if not path.exists():
            raise FileNotFoundError(f"OHLCV file not found for {t}: {path}")
        df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
        out[t] = df
    return out


def build_dataloaders(
    config: dict,
    sentiment_by_stock: dict[str, pd.Series] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train/val/test DataLoaders along with the fitted normalizer."""
    ohlcv = load_raw_ohlcv(config["data"]["ohlcv_dir"], config["data"]["stocks"])
    sentiment_by_stock = sentiment_by_stock or {t: None for t in ohlcv}

    train_end = config["data"]["train_period"][1]
    val_end = config["data"]["val_period"][1]

    features_train = {}
    features_val = {}
    features_test = {}
    normalizers: dict[str, ExpandingZScore] = {}

    for ticker, df in ohlcv.items():
        feats = build_prediction_features(df, sentiment_by_stock.get(ticker))
        train_slice = feats.loc[feats.index <= train_end]
        val_slice = feats.loc[(feats.index > train_end) & (feats.index <= val_end)]
        test_slice = feats.loc[feats.index > val_end]

        norm = ExpandingZScore().fit(train_slice)
        normalizers[ticker] = norm

        features_train[ticker] = norm.transform(train_slice, frozen=False)
        features_val[ticker] = norm.transform(val_slice, frozen=True)
        features_test[ticker] = norm.transform(test_slice, frozen=True)

    train_ds = OHLCVDataset(features_train, config["data"]["sequence_length"], tuple(config["data"]["prediction_horizons"]))
    val_ds = OHLCVDataset(features_val, config["data"]["sequence_length"], tuple(config["data"]["prediction_horizons"]))
    test_ds = OHLCVDataset(features_test, config["data"]["sequence_length"], tuple(config["data"]["prediction_horizons"]))

    train_loader = DataLoader(train_ds, batch_size=config["node_transformer"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["node_transformer"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["node_transformer"]["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader, {"normalizers": normalizers}
