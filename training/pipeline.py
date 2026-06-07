"""Four-stage training pipeline:

  Stage 1: train the autoencoder on stable-VIX days (eq. (4) loss).
  Stage 2: train the dual node transformers on regime-stratified data subsets.
  Stage 3: train the SAC controller with frozen prediction components.
  Stage 4: joint fine-tuning with a 10x learning-rate reduction.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import Autoencoder, DualPathwayFramework, NodeTransformer, SACController


class TrainingPipeline:
    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = device

        num_nodes = len(config["data"]["stocks"])
        ae_cfg = config["autoencoder"]
        nf_cfg = config["node_transformer"]

        self.autoencoder = Autoencoder(
            input_dim=config["data"]["prediction_feature_dim"],
            hidden_layers=tuple(ae_cfg["hidden_layers"]),
            latent_dim=ae_cfg["latent_dim"],
        ).to(device)

        self.normal_nf = NodeTransformer(
            feature_dim=config["data"]["prediction_feature_dim"],
            num_nodes=num_nodes,
            num_layers=nf_cfg["num_layers"],
            num_heads=nf_cfg["num_heads"],
            model_dim=nf_cfg["model_dim"],
            ffn_dim=nf_cfg["ffn_dim"],
            dropout=nf_cfg["dropout"],
            context_dim=0,
        ).to(device)

        self.event_nf = NodeTransformer(
            feature_dim=config["data"]["prediction_feature_dim"],
            num_nodes=num_nodes,
            num_layers=nf_cfg["num_layers"],
            num_heads=nf_cfg["num_heads"],
            model_dim=nf_cfg["model_dim"],
            ffn_dim=nf_cfg["ffn_dim"],
            dropout=nf_cfg["dropout"],
            context_dim=nf_cfg["context_dim"],
        ).to(device)

        self.framework = DualPathwayFramework(
            self.autoencoder, self.normal_nf, self.event_nf,
            initial_alpha=config["sac"]["initial_alpha"],
        )

        # SAC state = [e_t, e_bar_{t-k:t}, sigma_t, rmse_{t-1}, da_{t-1}, alpha_{t-1}, tau_{t-1}]
        sac_state_dim = 7
        self.sac = SACController(
            state_dim=sac_state_dim,
            action_dim=2,
            hidden_layers=tuple(config["sac"]["hidden_layers"]),
            lr=config["sac"]["learning_rate"],
            tau_soft=config["sac"]["soft_update_tau"],
            replay_capacity=config["sac"]["replay_buffer_size"],
            action_bounds=tuple(config["sac"]["action_bounds"]),
            device=device,
        )

    # --------------- Stage 1: Autoencoder ---------------
    def train_autoencoder(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        cfg = self.config["autoencoder"]
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=cfg["learning_rate"])
        best_val = float("inf")
        patience = cfg.get("early_stopping_patience", 5)
        bad_epochs = 0

        for epoch in range(cfg["epochs"]):
            self.autoencoder.train()
            for batch in tqdm(train_loader, desc=f"AE epoch {epoch + 1}/{cfg['epochs']}"):
                x = batch["features"][:, -1, :].to(self.device)  # last timestep features
                x_hat, _ = self.autoencoder(x)
                loss = torch.mean((x - x_hat) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = self._eval_ae_loss(val_loader)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    @torch.no_grad()
    def _eval_ae_loss(self, loader: DataLoader) -> float:
        self.autoencoder.eval()
        losses = []
        for batch in loader:
            x = batch["features"][:, -1, :].to(self.device)
            x_hat, _ = self.autoencoder(x)
            losses.append(torch.mean((x - x_hat) ** 2).item())
        return float(np.mean(losses))

    # --------------- Stage 2: Dual NodeFormers ---------------
    def train_dual_nodeformers(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        regime_threshold: float,
    ) -> None:
        cfg = self.config["node_transformer"]
        loss_cfg = self.config["loss"]

        opt_normal = torch.optim.Adam(self.normal_nf.parameters(), lr=cfg["learning_rate"])
        opt_event = torch.optim.Adam(self.event_nf.parameters(), lr=cfg["learning_rate"])

        bce = nn.BCEWithLogitsLoss()

        for epoch in range(cfg["epochs"]):
            self.normal_nf.train()
            self.event_nf.train()
            for batch in tqdm(train_loader, desc=f"NF epoch {epoch + 1}/{cfg['epochs']}"):
                features = batch["features"].to(self.device)
                stock_ids = batch["stock_id"].to(self.device)
                y_true = batch["y_1"].to(self.device)

                with torch.no_grad():
                    e_t = self.autoencoder.reconstruction_error(features[:, -1, :])
                    is_normal = e_t < regime_threshold

                if is_normal.any():
                    pred_normal = self.normal_nf(features[is_normal].unsqueeze(1), stock_ids[is_normal].unsqueeze(1))
                    target_normal = y_true[is_normal].unsqueeze(1)
                    loss_normal = self._composite_loss(pred_normal, target_normal, loss_cfg, bce)
                    opt_normal.zero_grad()
                    loss_normal.backward()
                    opt_normal.step()

                if (~is_normal).any():
                    ctx = torch.zeros((int((~is_normal).sum().item()), cfg["context_dim"]), device=self.device)
                    pred_event = self.event_nf(
                        features[~is_normal].unsqueeze(1),
                        stock_ids[~is_normal].unsqueeze(1),
                        context=ctx,
                    )
                    target_event = y_true[~is_normal].unsqueeze(1)
                    loss_event = self._composite_loss(pred_event, target_event, loss_cfg, bce)
                    opt_event.zero_grad()
                    loss_event.backward()
                    opt_event.step()

    def _composite_loss(self, pred: torch.Tensor, target: torch.Tensor, loss_cfg: dict, bce: nn.Module) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        direction_logits = pred - target.detach()
        direction_target = (target > 0).float()
        dir_loss = bce(direction_logits, direction_target)
        reg = sum(p.pow(2.0).sum() for p in self.normal_nf.parameters()) * loss_cfg["lambda_reg"]
        return loss_cfg["lambda_mse"] * mse + loss_cfg["lambda_dir"] * dir_loss + reg

    # --------------- Stage 3: SAC Controller ---------------
    def train_sac(self, train_loader: DataLoader, initial_tau: float) -> None:
        cfg = self.config["sac"]
        self.framework.set_routing_parameters(initial_tau, cfg["initial_alpha"])

        rmse_prev = 0.0
        da_prev = 0.0
        tau = initial_tau
        alpha = cfg["initial_alpha"]
        history = []

        for epoch in range(cfg["epochs"]):
            for batch in tqdm(train_loader, desc=f"SAC epoch {epoch + 1}/{cfg['epochs']}"):
                features = batch["features"].to(self.device)
                stock_ids = batch["stock_id"].to(self.device)
                y_true = batch["y_1"].to(self.device)

                with torch.no_grad():
                    e_t = self.autoencoder.reconstruction_error(features[:, -1, :]).mean().item()
                    sigma_t = features[:, -1, -2].abs().mean().item()  # rolling vol proxy
                    e_bar = float(np.mean(history[-cfg["state_history_window"]:])) if history else e_t
                    history.append(e_t)

                state = np.array([e_t, e_bar, sigma_t, rmse_prev, da_prev, alpha, tau], dtype=np.float32)
                delta = self.sac.act(state)
                tau = float(np.clip(tau + delta[0], 0.0, 5.0))
                alpha = float(np.clip(alpha + delta[1], 0.0, 1.0))
                self.framework.set_routing_parameters(tau, alpha)

                with torch.no_grad():
                    ctx = torch.zeros((features.size(0), self.config["node_transformer"]["context_dim"]), device=self.device)
                    out = self.framework(
                        features.unsqueeze(1),
                        stock_ids.unsqueeze(1),
                        x_router=features[:, -1, :],
                        context=ctx,
                    )
                    pred = out["blended_prediction"].squeeze()
                    rmse_t = float(torch.sqrt(torch.mean((pred - y_true) ** 2)).item())
                    da_t = float(((pred > 0) == (y_true > 0)).float().mean().item())

                reward = (
                    -rmse_t
                    - cfg["reward"]["lambda_dir"] * (1.0 - da_t)
                    - cfg["reward"]["lambda_stable"] * abs(delta[0])
                )

                next_state = np.array([e_t, e_bar, sigma_t, rmse_t, da_t, alpha, tau], dtype=np.float32)
                self.sac.replay.push(
                    type("T", (), {"state": state, "action": delta, "reward": reward, "next_state": next_state, "done": 0.0})()
                )
                self.sac.update(batch_size=cfg["batch_size"])

                rmse_prev, da_prev = rmse_t, da_t

    # --------------- Stage 4: Joint fine-tuning ---------------
    def fine_tune(self, train_loader: DataLoader) -> None:
        cfg = self.config["finetune"]
        opt_ae = torch.optim.Adam(self.autoencoder.parameters(), lr=cfg["learning_rates"]["autoencoder"])
        opt_nn = torch.optim.Adam(self.normal_nf.parameters(), lr=cfg["learning_rates"]["node_transformer"])
        opt_en = torch.optim.Adam(self.event_nf.parameters(), lr=cfg["learning_rates"]["node_transformer"])

        for epoch in range(cfg["epochs"]):
            for batch in tqdm(train_loader, desc=f"FT epoch {epoch + 1}/{cfg['epochs']}"):
                features = batch["features"].to(self.device)
                stock_ids = batch["stock_id"].to(self.device)
                y_true = batch["y_1"].to(self.device)
                ctx = torch.zeros((features.size(0), self.config["node_transformer"]["context_dim"]), device=self.device)

                out = self.framework(
                    features.unsqueeze(1),
                    stock_ids.unsqueeze(1),
                    x_router=features[:, -1, :],
                    context=ctx,
                )
                loss = torch.mean((out["blended_prediction"].squeeze() - y_true) ** 2)

                opt_ae.zero_grad()
                opt_nn.zero_grad()
                opt_en.zero_grad()
                loss.backward()
                opt_ae.step()
                opt_nn.step()
                opt_en.step()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "autoencoder": self.autoencoder.state_dict(),
                "normal_nf": self.normal_nf.state_dict(),
                "event_nf": self.event_nf.state_dict(),
                "sac_actor": self.sac.actor.state_dict(),
                "sac_critic": self.sac.critic.state_dict(),
                "tau": float(self.framework.tau.item()),
                "alpha": float(self.framework.alpha.item()),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(ckpt["autoencoder"])
        self.normal_nf.load_state_dict(ckpt["normal_nf"])
        self.event_nf.load_state_dict(ckpt["event_nf"])
        self.sac.actor.load_state_dict(ckpt["sac_actor"])
        self.sac.critic.load_state_dict(ckpt["sac_critic"])
        self.framework.set_routing_parameters(ckpt["tau"], ckpt["alpha"])
        self.sac.freeze()
