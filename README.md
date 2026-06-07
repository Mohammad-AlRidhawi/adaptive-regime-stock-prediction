# Adaptive Regime-Aware Stock Price Prediction

PyTorch implementation of the adaptive regime-aware stock price prediction framework presented in:

> Al Ridhawi M., Haj Ali M., and Al Osman H. *Adaptive Regime-Aware Stock Price Prediction Using Autoencoder-Gated Dual Node Transformers with Reinforcement Learning Control.* Submitted to *Applied Intelligence*, 2026.

The framework combines three components inside a closed-loop system:
1. An **autoencoder** trained on stable-VIX days that produces a reconstruction-error anomaly score.
2. **Dual node transformer pathways** with independent weights, one specialised for stable conditions and one for event-driven conditions.
3. A **Soft Actor-Critic (SAC) controller** that adapts the routing threshold and the pathway blending weight from realised prediction performance.

## Repository structure

```
.
├── configs/                YAML configs for each component and training stage
├── data/                   Data loaders, feature engineering, normalization, sentiment
├── models/                 Autoencoder, node transformer, dual pathway, SAC actor/critics
├── training/               Multi-stage training pipeline (Stages 1-4)
├── evaluation/             MAPE/RMSE/DA/Theil's U/CTR metrics and statistical tests
├── backtest/               Top-K trading strategy and risk-adjusted performance metrics
├── utils/                  Logging, checkpoints, reproducibility seeding
├── scripts/                CLI entry points for training, evaluation, backtesting
├── tests/                  Unit tests for data, models, metrics
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

Tested on Python 3.10 with CUDA 12.1.

```bash
git clone git@github.com:Mohammad-AlRidhawi/adaptive-regime-stock-prediction.git
cd adaptive-regime-stock-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

The framework expects two complementary data streams:

- **Financial market data (FMD)**: daily OHLCV for 20 S&P 500 stocks from January 1982 to March 2025 (Yahoo Finance via `yfinance`).
- **Sentiment data**: daily aggregated sentiment scores per stock, produced by a fine-tuned BERT classifier applied to the Comprehensive Stock Sentiment (CSS) dataset introduced in our prior work (`alridhawi2025nodeformer`).

Place raw OHLCV CSV files under `data/raw/ohlcv/` (one CSV per ticker) and the daily sentiment matrix under `data/raw/sentiment/css_daily_scores.parquet`. The data loaders handle the rest (technical indicators, expanding-window z-score, train/validation/test splits).

## Quick start

Train the full pipeline:

```bash
python scripts/train.py --config configs/default.yaml
```

Evaluate a trained model on the test set:

```bash
python scripts/evaluate.py --checkpoint runs/latest/checkpoints/final.pt --horizon 1
```

Run the trading backtest (long-only top-K):

```bash
python scripts/backtest.py --checkpoint runs/latest/checkpoints/final.pt --top-k 5 --cost-bps 10
```

## Reproducing paper results

To reproduce Tables 3-9 and Figures 11-13 of the paper, run:

```bash
bash scripts/reproduce_paper.sh
```

This will sequentially train each of the 14 baselines plus the two proposed variants, evaluate at 1/5/20-day horizons, run the trading backtest, and generate all PGF figures used in the manuscript. Total wall-clock time on a single NVIDIA A100 40 GB GPU is approximately 22 hours.

## Hardware

Tested on Google Cloud Platform `a2-highgpu-1g` (1x NVIDIA A100 40 GB, 12 vCPUs, 85 GB RAM). The framework also runs on consumer-grade hardware (e.g., RTX 3090/4090) with the same configuration; training time scales with GPU FLOPs.

## License

Released under the MIT License. See `LICENSE`.

## Citation

```bibtex
@article{alridhawi2026adaptive,
  title={Adaptive Regime-Aware Stock Price Prediction Using Autoencoder-Gated Dual Node Transformers with Reinforcement Learning Control},
  author={Al Ridhawi, Mohammad and Haj Ali, Mahtab and Al Osman, Hussein},
  journal={Applied Intelligence},
  year={2026},
  note={Under review}
}
```

## Access

This repository is private during the peer-review process. Code, configs, and pretrained checkpoints are available upon reasonable request from the corresponding author (malri039@uottawa.ca).

## Contact

Mohammad Al Ridhawi - malri039@uottawa.ca
School of Electrical Engineering and Computer Science, University of Ottawa
