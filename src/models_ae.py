"""Autoencoder model and training utilities."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


class AutoEncoder(nn.Module):
    """Symmetric autoencoder with LayerNorm, GELU, and Dropout."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = cfg.AE_DROPOUT,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = cfg.AE_HIDDEN_DIMS  # [512, 128]

        # Build encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder (mirror)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    latent_dim: int = cfg.LATENT_DIM,
    device: str = cfg.DEVICE,
    lr: float = cfg.AE_LR,
    weight_decay: float = cfg.AE_WEIGHT_DECAY,
    batch_size: int = cfg.AE_BATCH_SIZE,
    max_epochs: int = cfg.AE_MAX_EPOCHS,
    patience: int = cfg.AE_PATIENCE,
    dropout: float = cfg.AE_DROPOUT,
    print_every: int = 25,
) -> tuple[AutoEncoder, int, list[float], list[float]]:
    """Train AE with early stopping and LR scheduling.

    Returns (model, best_epoch, train_losses, val_losses).
    """
    input_dim = X_train.shape[1]
    model = AutoEncoder(input_dim, latent_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.AE_SCHEDULER_FACTOR,
        patience=cfg.AE_SCHEDULER_PATIENCE,
        min_lr=1e-6,
    )
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(max_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(X_train))

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_t), val_t))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch [{epoch+1:>4d}/{max_epochs}] "
                f"train={train_losses[-1]:.4f}  val={val_loss:.4f}  lr={lr_now:.1e}"
            )

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best={best_epoch+1})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, train_losses, val_losses


@torch.no_grad()
def encode_np(model: AutoEncoder, X: np.ndarray, device: str = cfg.DEVICE) -> np.ndarray:
    """Encode numpy array to latent space."""
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=device)
    return model.encode(x).cpu().numpy().astype(np.float32)


@torch.no_grad()
def decode_np(model: AutoEncoder, Z: np.ndarray, device: str = cfg.DEVICE) -> np.ndarray:
    """Decode latent numpy array back to input space."""
    model.eval()
    z = torch.tensor(Z, dtype=torch.float32, device=device)
    return model.decode(z).cpu().numpy().astype(np.float32)


@torch.no_grad()
def recon_mse(model: AutoEncoder, X: np.ndarray, device: str = cfg.DEVICE) -> float:
    """Compute reconstruction MSE on a numpy array."""
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=device)
    x_hat = model(x).cpu().numpy()
    return float(np.mean((X - x_hat) ** 2))
