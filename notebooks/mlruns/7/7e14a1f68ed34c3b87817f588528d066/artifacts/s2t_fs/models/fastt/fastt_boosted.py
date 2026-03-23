"""
FASTT-SDT: Differentiable path (Algorithm 2).

Boosted soft decision trees with per-round learnable feature transforms,
trained end-to-end via backpropagation on the expected-WER objective.

Each boosting round b learns:
  - A feature transform  Tθb(·)
  - A soft decision tree  hb(·) producing L logits

Final output:  q_i = Σ_b  hb(Tθb(z_i))
"""

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from s2t_fs.models.adastt_mlp import AudioFeatureDataset
from s2t_fs.models.fastt.transforms import build_transform
from s2t_fs.models.sdtr_models import _SDTR


class _FASTTBoostedNet(nn.Module):
    """Boosted SDT ensemble with per-round feature transforms."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_rounds: int = 3,
        transform_type: str = "diagonal",
        transform_kwargs: dict | None = None,
        num_trees: int = 10,
        depth: int = 4,
        lmbda: float = 0.1,
        lmbda2: float = 0.01,
    ):
        super().__init__()
        self.num_rounds = num_rounds
        t_kwargs = transform_kwargs or {}

        self.transforms = nn.ModuleList()
        self.trees = nn.ModuleList()

        for _ in range(num_rounds):
            transform = build_transform(
                transform_type, in_features=in_features, **t_kwargs
            )

            out_dim = in_features
            if hasattr(transform, "linear"):
                out_dim = transform.linear.out_features
            elif hasattr(transform, "w2"):
                out_dim = transform.w2.out_features

            tree = _SDTR(
                in_features=out_dim,
                num_trees=num_trees,
                depth=depth,
                tree_dim=num_classes,
                flatten_output=True,
                lmbda=lmbda,
                lmbda2=lmbda2,
            )

            self.transforms.append(transform)
            self.trees.append(tree)

    def forward(self, z: torch.Tensor):
        total_logits = None
        total_reg = torch.tensor(0.0, device=z.device)
        total_transform_reg = torch.tensor(0.0, device=z.device)

        for transform, tree in zip(self.transforms, self.trees):
            u = transform(z)
            logits, reg_loss, l1_loss = tree(u)

            total_logits = logits if total_logits is None else total_logits + logits

            if reg_loss is not None:
                total_reg = total_reg + reg_loss
            if l1_loss is not None:
                total_reg = total_reg + l1_loss

            total_transform_reg = total_transform_reg + transform.regularization_loss()

        return total_logits, total_reg, total_transform_reg


class _FASTTBoostedLoss(nn.Module):
    """Expected-WER loss with SDT regularization and transform regularization."""

    def forward(self, model_output, wer_targets):
        logits, sdt_reg, transform_reg = model_output
        wer_targets = wer_targets.reshape_as(logits)

        probs = torch.softmax(logits, dim=1)
        expected_wer = (probs * wer_targets).sum(dim=1).mean()

        sdt_reg = torch.nan_to_num(sdt_reg, nan=0.0, posinf=0.0, neginf=0.0)
        transform_reg = torch.nan_to_num(transform_reg, nan=0.0, posinf=0.0, neginf=0.0)

        return expected_wer + sdt_reg + transform_reg


class FASTTBoosted(BaseEstimator, ClassifierMixin):
    """FASTT for differentiable selectors (Algorithm 2).

    Boosted ensemble of soft decision trees with per-round learnable feature
    transforms, trained end-to-end on the expected-WER objective.

    Parameters
    ----------
    num_rounds : int
        Number of boosting rounds B (each with its own transform + SDT).
    transform_type : str
        Feature transform variant: 'identity', 'diagonal', 'linear',
        'low_rank', or 'nonlinear'.
    transform_kwargs : dict or None
        Extra arguments for the transform (e.g., lambda1, lambda2, bottleneck_dim).
    num_trees : int
        Number of trees per SDT ensemble at each round.
    depth : int
        Depth of each soft decision tree.
    lmbda : float
        SDT routing regularization strength.
    lr : float
        Learning rate for AdamW optimizer.
    weight_decay : float
        Weight decay for AdamW optimizer.
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        num_rounds=3,
        transform_type="diagonal",
        transform_kwargs=None,
        num_trees=10,
        depth=4,
        lmbda=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
        epochs=100,
        patience=20,
        random_state=42,
    ):
        self.num_rounds = num_rounds
        self.transform_type = transform_type
        self.transform_kwargs = transform_kwargs
        self.num_trees = num_trees
        self.depth = depth
        self.lmbda = lmbda
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.num_classes_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.num_classes_ = y.shape[1]

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        train_loader = DataLoader(
            AudioFeatureDataset(X_train, Y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            AudioFeatureDataset(X_val, Y_val),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model_ = _FASTTBoostedNet(
            in_features=X.shape[1],
            num_classes=self.num_classes_,
            num_rounds=self.num_rounds,
            transform_type=self.transform_type,
            transform_kwargs=self.transform_kwargs,
            num_trees=self.num_trees,
            depth=self.depth,
            lmbda=self.lmbda,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )
        loss_fn = _FASTTBoostedLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for _ in range(self.epochs):
            self.model_.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(x_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

            self.model_.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    val_loss_accum += (
                        loss_fn(self.model_(x_batch), y_batch).item() * x_batch.size(0)
                    )

            avg_val = val_loss_accum / len(X_val)
            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _get_logits(self, X):
        z = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            logits, _, _ = self.model_(z)
        return logits

    def predict_proba(self, X):
        return torch.softmax(self._get_logits(X), dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        """Negative mean WER (higher is better, sklearn convention)."""
        preds = self.predict(X)
        return -y[np.arange(len(y)), preds].mean()

    def get_feature_importances(self):
        """Return per-round feature importance derived from the transform layers."""
        if self.model_ is None:
            return None
        weights = []
        for transform in self.model_.transforms:
            if self.transform_type == "diagonal":
                weights.append(np.abs(transform.q.detach().cpu().numpy()))
            elif self.transform_type == "linear":
                weights.append(torch.norm(transform.linear.weight, p=2, dim=0).detach().cpu().numpy())
            elif self.transform_type in ["low_rank", "nonlinear"]:
                weights.append(torch.norm(transform.w1.weight, p=2, dim=0).detach().cpu().numpy())
            elif self.transform_type == "identity":
                continue
        return weights if weights else None
