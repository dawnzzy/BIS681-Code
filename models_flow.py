"""Flow matching model and training/sampling utilities."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from src.preprocess import set_seed


class FlowNet(nn.Module):
    """MLP that predicts velocity v(t, x_t) for flow matching."""

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim + 1, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Predict velocity given time t (B,1) and state x_t (B,dim)."""
        return self.net(torch.cat([t, x_t], dim=-1))

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        dim: int,
        device: str,
        n_steps: int = cfg.FM_SAMPLE_STEPS,
    ) -> np.ndarray:
        """Generate samples via midpoint integration from t=0 (noise) to t=1 (data)."""
        self.eval()
        x = torch.randn(n_samples, dim, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_i = torch.full((n_samples, 1), i * dt, device=device)
            t_mid = torch.full((n_samples, 1), (i + 0.5) * dt, device=device)
            # Midpoint method (2nd-order)
            v_i = self.forward(t_i, x)
            x_mid = x + v_i * (dt / 2)
            v_mid = self.forward(t_mid, x_mid)
            x = x + v_mid * dt

        return x.cpu().numpy().astype(np.float32)


class ConditionalFlowNet(nn.Module):
    """MLP that predicts velocity v(t, x_t, label) for conditional flow matching."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        self.n_classes = n_classes
        # Input: [t (1), x_t (dim), label_onehot (n_classes)]
        layers: list[nn.Module] = [nn.Linear(dim + 1 + n_classes, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x_t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Predict velocity given time t (B,1), state x_t (B,dim), label one-hot (B,n_classes)."""
        return self.net(torch.cat([t, x_t, label], dim=-1))

    @torch.no_grad()
    def sample(
        self,
        n_per_class: list[int],
        dim: int,
        device: str,
        n_steps: int = cfg.FM_SAMPLE_STEPS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate samples conditioned on class labels.

        Parameters
        ----------
        n_per_class : list[int]
            Number of samples to generate for each class (index = class id).
        dim : int
            Dimensionality of the data space.
        device : str
            Device to run on.
        n_steps : int
            Number of integration steps.

        Returns
        -------
        samples : np.ndarray, shape (sum(n_per_class), dim)
        labels : np.ndarray, shape (sum(n_per_class),)
        """
        self.eval()
        all_samples = []
        all_labels = []

        for cls_id, n in enumerate(n_per_class):
            if n == 0:
                continue
            x = torch.randn(n, dim, device=device)
            label = torch.zeros(n, self.n_classes, device=device)
            label[:, cls_id] = 1.0  # one-hot
            dt = 1.0 / n_steps

            for i in range(n_steps):
                t_i = torch.full((n, 1), i * dt, device=device)
                t_mid = torch.full((n, 1), (i + 0.5) * dt, device=device)
                v_i = self.forward(t_i, x, label)
                x_mid = x + v_i * (dt / 2)
                v_mid = self.forward(t_mid, x_mid, label)
                x = x + v_mid * dt

            all_samples.append(x.cpu().numpy().astype(np.float32))
            all_labels.append(np.full(n, cls_id, dtype=np.int64))

        return np.concatenate(all_samples, axis=0), np.concatenate(all_labels, axis=0)


def train_conditional_flow_matching(
    Z_train: np.ndarray,
    labels_train: np.ndarray,
    dim: int,
    n_classes: int,
    device: str = cfg.DEVICE,
    hidden: int = 256,
    n_layers: int = 4,
    lr: float = 1e-3,
    batch_size: int = 256,
    n_epochs: int = 500,
    print_every: int = 50,
) -> tuple[ConditionalFlowNet, list[float]]:
    """Train a conditional flow matching model. Returns (model, loss_history)."""
    set_seed()
    model = ConditionalFlowNet(dim=dim, n_classes=n_classes, hidden=hidden, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # One-hot encode labels
    labels_onehot = np.zeros((len(labels_train), n_classes), dtype=np.float32)
    labels_onehot[np.arange(len(labels_train)), labels_train] = 1.0

    dataset = TensorDataset(
        torch.tensor(Z_train, dtype=torch.float32),
        torch.tensor(labels_onehot, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history: list[float] = []
    model.train()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for x1, lab in loader:
            x1 = x1.to(device)
            lab = lab.to(device)
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), 1, device=device)

            x_t = (1 - t) * x0 + t * x1
            v_target = x1 - x0

            optimizer.zero_grad()
            v_pred = model(t, x_t, lab)
            loss = loss_fn(v_pred, v_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(x1)

        avg_loss = epoch_loss / len(Z_train)
        loss_history.append(avg_loss)

        if epoch % print_every == 0 or epoch == 1:
            print(f"  Epoch [{epoch:>4d}/{n_epochs}] loss={avg_loss:.4f}")

    model.eval()
    return model, loss_history


def train_flow_matching(
    Z_train: np.ndarray,
    dim: int,
    device: str = cfg.DEVICE,
    hidden: int = 256,
    n_layers: int = 4,
    lr: float = 1e-3,
    batch_size: int = 256,
    n_epochs: int = 500,
    print_every: int = 50,
) -> tuple[FlowNet, list[float]]:
    """Train a flow matching model. Returns (model, loss_history)."""
    set_seed()
    model = FlowNet(dim=dim, hidden=hidden, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(Z_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history: list[float] = []
    model.train()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for (x1,) in loader:
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)  # noise prior
            t = torch.rand(len(x1), 1, device=device)

            # Interpolation and target velocity (optimal transport)
            x_t = (1 - t) * x0 + t * x1
            v_target = x1 - x0

            optimizer.zero_grad()
            v_pred = model(t, x_t)
            loss = loss_fn(v_pred, v_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(x1)

        avg_loss = epoch_loss / len(Z_train)
        loss_history.append(avg_loss)

        if epoch % print_every == 0 or epoch == 1:
            print(f"  Epoch [{epoch:>4d}/{n_epochs}] loss={avg_loss:.4f}")

    model.eval()
    return model, loss_history
