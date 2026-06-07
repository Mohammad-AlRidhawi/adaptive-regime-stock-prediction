"""End-to-end training entry point: runs Stages 1 through 4."""

import argparse
from pathlib import Path

import yaml

from data import SentimentLoader, build_dataloaders
from training import TrainingPipeline
from utils import get_logger, set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_global_seed(config["seed"])
    logger = get_logger("train")

    sentiment_loader = SentimentLoader(config["data"]["sentiment_path"])
    sentiment_by_stock = {t: sentiment_loader.get_series(t) for t in config["data"]["stocks"]}

    train_loader, val_loader, test_loader, _ = build_dataloaders(config, sentiment_by_stock)

    pipeline = TrainingPipeline(config, device=config["device"])

    logger.info("Stage 1: training autoencoder on stable-VIX days")
    pipeline.train_autoencoder(train_loader, val_loader)

    logger.info("Stage 2: training dual node transformers on regime-stratified subsets")
    # Initialize the routing threshold at the 95th percentile of training reconstruction errors.
    import torch

    with torch.no_grad():
        errs = []
        for batch in train_loader:
            x = batch["features"][:, -1, :].to(config["device"])
            errs.append(pipeline.autoencoder.reconstruction_error(x).cpu())
        errs = torch.cat(errs)
        tau0 = float(torch.quantile(errs, 0.95).item())
    logger.info(f"Initial routing threshold tau = {tau0:.4f}")
    pipeline.train_dual_nodeformers(train_loader, val_loader, regime_threshold=tau0)

    logger.info("Stage 3: training SAC controller against prediction-quality reward")
    pipeline.train_sac(train_loader, initial_tau=tau0)

    logger.info("Stage 4: joint fine-tuning with 10x learning-rate reduction")
    pipeline.fine_tune(train_loader)

    out_dir = Path(config["output_dir"]) / "checkpoints"
    pipeline.save(out_dir / "final.pt")
    logger.info(f"Saved final checkpoint to {out_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
