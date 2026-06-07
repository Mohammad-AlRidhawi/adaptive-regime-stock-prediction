"""Trading backtest entry point: long-only top-K with transaction costs."""

import argparse

import numpy as np
import pandas as pd
import torch
import yaml

from backtest import LongOnlyTopK, compute_backtest_metrics
from data import SentimentLoader, build_dataloaders
from training import TrainingPipeline
from utils import get_logger, set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_global_seed(config["seed"])
    logger = get_logger("backtest")

    sentiment_loader = SentimentLoader(config["data"]["sentiment_path"])
    sentiment_by_stock = {t: sentiment_loader.get_series(t) for t in config["data"]["stocks"]}

    _, _, test_loader, _ = build_dataloaders(config, sentiment_by_stock)

    pipeline = TrainingPipeline(config, device=config["device"])
    pipeline.load(args.checkpoint)
    pipeline.framework.eval()

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(config["device"])
            stock_ids = batch["stock_id"].to(config["device"])
            ctx = torch.zeros((features.size(0), config["node_transformer"]["context_dim"]), device=config["device"])
            out = pipeline.framework(
                features.unsqueeze(1), stock_ids.unsqueeze(1),
                x_router=features[:, -1, :], context=ctx,
            )
            for i in range(features.size(0)):
                rows.append(
                    {
                        "stock_id": int(stock_ids[i].item()),
                        "predicted_return": float(out["blended_prediction"][i].item()),
                        "actual_return": float(batch["y_1"][i].item()),
                    }
                )

    df = pd.DataFrame(rows)
    tickers = config["data"]["stocks"]
    predictions = df.pivot(columns="stock_id", values="predicted_return")
    predictions.columns = [tickers[i] for i in predictions.columns]
    actual = df.pivot(columns="stock_id", values="actual_return")
    actual.columns = [tickers[i] for i in actual.columns]

    strategy = LongOnlyTopK(top_k=args.top_k, cost_bps=args.cost_bps)
    output = strategy.run(predictions, actual)
    metrics = compute_backtest_metrics(output, risk_free_rate=config["evaluation"]["risk_free_rate"])

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
