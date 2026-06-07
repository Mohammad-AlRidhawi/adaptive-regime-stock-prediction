"""Evaluation entry point: loads a trained checkpoint and reports metrics on the test set."""

import argparse

import numpy as np
import torch
import yaml

from data import SentimentLoader, build_dataloaders
from evaluation import compute_all_metrics
from training import TrainingPipeline
from utils import get_logger, set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_global_seed(config["seed"])
    logger = get_logger("evaluate")

    sentiment_loader = SentimentLoader(config["data"]["sentiment_path"])
    sentiment_by_stock = {t: sentiment_loader.get_series(t) for t in config["data"]["stocks"]}

    _, _, test_loader, _ = build_dataloaders(config, sentiment_by_stock)

    pipeline = TrainingPipeline(config, device=config["device"])
    pipeline.load(args.checkpoint)
    pipeline.framework.eval()

    preds, targets, prevs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(config["device"])
            stock_ids = batch["stock_id"].to(config["device"])
            ctx = torch.zeros((features.size(0), config["node_transformer"]["context_dim"]), device=config["device"])
            out = pipeline.framework(
                features.unsqueeze(1), stock_ids.unsqueeze(1),
                x_router=features[:, -1, :], context=ctx,
            )
            y_hat = out["blended_prediction"].squeeze().cpu().numpy()
            y = batch[f"y_{args.horizon}"].numpy()
            y_prev = features[:, -1, 3].cpu().numpy()  # close at last input timestep
            preds.append(y_hat)
            targets.append(y)
            prevs.append(y_prev)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    prevs = np.concatenate(prevs)

    metrics = compute_all_metrics(targets, preds, prevs)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
