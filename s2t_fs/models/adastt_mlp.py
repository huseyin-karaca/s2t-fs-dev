import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from s2t_fs.utils.torch_utils import get_torch_device, seed_device


class AudioFeatureDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class AdasttMLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(AdasttMLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def expected_wer_loss(logits, wer_targets):
    probs = torch.softmax(logits, dim=1)
    return torch.sum(probs * wer_targets, dim=1).mean()


class AdaSTTMLP(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=1024,
        dropout_rate=0.20,
        epochs=100,
        patience=20,
        random_state=42,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = get_torch_device()
        self.model_ = None
        self.scaler_ = None
        self.num_classes_ = None

    def fit(self, X, y):
        # Set seeds for reproducibility
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        seed_device(self.device, self.random_state)

        self.num_classes_ = y.shape[1]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state
        )

        train_loader = DataLoader(
            AudioFeatureDataset(X_train, Y_train), batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            AudioFeatureDataset(X_val, Y_val), batch_size=self.batch_size, shuffle=False
        )

        input_dim = X.shape[1]
        self.model_ = AdasttMLPNet(input_dim, self.num_classes_, self.dropout_rate).to(self.device)

        optimizer = optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        use_amp = self.device.type == "cuda"
        if use_amp:
            scaler_amp = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.model_.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model_(x_batch)
                        loss = expected_wer_loss(logits, y_batch)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    logits = self.model_(x_batch)
                    loss = expected_wer_loss(logits, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                    optimizer.step()

            self.model_.eval()
            val_loss_accum = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    logits = self.model_(x_batch)
                    loss = expected_wer_loss(logits, y_batch)
                    val_loss_accum += loss.item() * x_batch.size(0)

            avg_val_loss = val_loss_accum / len(X_val)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model_.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)

        return self

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        """
        Returns negative Mean Word Error Rate (WER).
        Sklearn meta-estimators expect 'greater is better'.
        """
        preds = self.predict(X)
        actual_wers = y[np.arange(len(y)), preds]
        return -actual_wers.mean()


class _MLPDifferentiable(nn.Module):
    """Bridge for AdasttMLPNet to return (logits, reg_loss, l1_loss) required by _FASTTBoostedNet."""
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.mlp = AdasttMLPNet(input_dim, output_dim, dropout_rate)
        
    def forward(self, x):
        # Return logits, reg_loss=None, l1_loss=None
        return self.mlp(x), None, None
