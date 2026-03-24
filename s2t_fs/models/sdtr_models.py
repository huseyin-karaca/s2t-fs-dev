"""
Soft Decision Tree with Regularisation (SDTR) — all components in one file.

Public API
----------
SingleSDTR   – sklearn-compatible wrapper around a single SDTR ensemble
BoostedSDTR  – sklearn-compatible wrapper around a boosted stack of SDTRs
"""

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from s2t_fs.models.adastt_mlp import AudioFeatureDataset
from s2t_fs.utils.torch_utils import get_torch_device, seed_device

# ── Low-level PyTorch building blocks ─────────────────────────────────────


class _ModuleWithInit(nn.Module):
    """Base class: data-aware lazy initializer triggered on the first forward pass."""

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(
            torch.tensor(0, dtype=torch.uint8), requires_grad=False
        )
        self._is_initialized_bool = None

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)


class _SDTR(_ModuleWithInit):
    """Soft Decision Tree with Regularisation (single ensemble)."""

    def __init__(
        self,
        in_features,
        num_trees,
        depth=6,
        tree_dim=1,
        flatten_output=True,
        init_func="zero",
        hidden_dim=256,
        lmbda=0.1,
        lmbda2=0.01,
        **kwargs,
    ):
        super().__init__()
        self.depth = depth
        self.num_trees = num_trees
        self.tree_dim = tree_dim
        self.flatten_output = flatten_output
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        self.init_func = init_func

        self.response = nn.Parameter(
            torch.zeros([num_trees, tree_dim, 2**depth]), requires_grad=True
        )

        if hidden_dim:
            self.input_fc = nn.Linear(in_features, hidden_dim)

        beta = 1.5
        self.feature_logit_layers = nn.ModuleList()
        self.betas = nn.ParameterList()
        for cur_depth in range(depth):
            layer = nn.Linear(in_features, num_trees * (2**cur_depth))
            nn.init.xavier_uniform_(layer.weight)
            self.feature_logit_layers.append(layer)
            self.betas.append(
                nn.Parameter(
                    torch.FloatTensor(np.full((num_trees, 2**cur_depth), beta)),
                    requires_grad=True,
                )
            )

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)

        batch_size = input.size(0)
        device = next(self.parameters()).device

        reg_loss = 0.0
        l1_loss = None
        reweighting = 1

        path_prob = Variable(torch.ones(batch_size, self.num_trees, 1), requires_grad=True).to(
            device
        )

        for cur_depth in range(len(self.feature_logit_layers)):
            logit = self.feature_logit_layers[cur_depth](input).view(
                batch_size, self.num_trees, -1
            )
            for param in self.feature_logit_layers[cur_depth].parameters():
                l1_term = param.norm(1) / self.num_trees * reweighting * self.lmbda2
                l1_loss = l1_term if l1_loss is None else l1_loss + l1_term
            reweighting /= 2

            prob = torch.sigmoid(torch.einsum("bnc,nc->bnc", logit, self.betas[cur_depth]))
            prob_right = 1 - prob

            penalty = torch.einsum("bnc,bnc->nc", prob, path_prob) / (
                torch.einsum("ijk->jk", path_prob) + 1e-4
            )
            reg_loss -= (
                self.lmbda
                * 2 ** (-cur_depth)
                * 0.5
                * torch.mean(torch.log(penalty) + torch.log(1 - penalty))
            )

            prob = prob * path_prob
            prob_right = prob_right * path_prob
            path_prob = torch.stack((prob.unsqueeze(-1), prob_right.unsqueeze(-1)), dim=3).view(
                batch_size, self.num_trees, 2 ** (cur_depth + 1)
            )

        response = torch.einsum("bnd,ncd->bnc", path_prob, self.response)
        response = torch.sum(response, dim=1).squeeze(-1)
        if self.flatten_output and response.dim() > 2:
            response = response.flatten(1, 2)
        return response, reg_loss, l1_loss

    def initialize(self, input, eps=1e-6):
        if self.init_func == "uniform":
            nn.init.uniform_(self.response, -1, 1)
        elif self.init_func == "xuniform":
            nn.init.xavier_uniform_(self.response)
        elif self.init_func == "normal":
            nn.init.normal_(self.response)
        # "zero" → no-op (already zeros)


class _BoostedSDTRNet(nn.Module):
    """Stack of SDTR ensembles whose outputs are summed (boosting-style)."""

    def __init__(
        self,
        in_features,
        num_boosting_layers=3,
        transform_type="identity",
        transform_hidden_dim=64,
        **sdt_params,
    ):
        super().__init__()
        self.transformers = nn.ModuleList()
        self.sdtrs = nn.ModuleList()

        for _ in range(num_boosting_layers):
            if transform_type == "identity":
                transformer = nn.Identity()
            elif transform_type == "linear":
                transformer = nn.Linear(in_features, in_features, bias=False)
            elif transform_type == "mlp":
                transformer = nn.Sequential(
                    nn.Linear(in_features, transform_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(transform_hidden_dim, in_features),
                )
            else:
                raise ValueError(f"Unknown transform_type: {transform_type}")

            self.transformers.append(transformer)
            self.sdtrs.append(_SDTR(in_features=in_features, **sdt_params))

    def forward(self, X):
        total, total_reg, total_l1 = None, 0.0, 0.0
        for transformer, sdtr in zip(self.transformers, self.sdtrs):
            pred, reg, l1 = sdtr(transformer(X))
            total = pred if total is None else total + pred
            if reg is not None:
                total_reg += reg
            if l1 is not None:
                total_l1 += l1
        return total, total_reg, total_l1


# ── Loss ───────────────────────────────────────────────────────────────────


class _SDTRLoss(nn.Module):
    """E[WER] loss plus the SDTR regularisation terms."""

    def forward(self, y_pred, y_true):
        response, reg_loss, l1_loss = y_pred
        y_true = y_true.reshape_as(response)
        probs = torch.softmax(response, dim=1)
        loss = (probs * y_true).sum(dim=1).mean()
        reg_loss = torch.nan_to_num(reg_loss, nan=0.0, posinf=0.0, neginf=0.0)
        l1_loss = torch.nan_to_num(l1_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss + reg_loss + l1_loss


# ── Sklearn-compatible base wrapper ────────────────────────────────────────


class _BaseSDTRWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
        epochs=100,
        patience=20,
        random_state=42,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = get_torch_device()
        self.model_ = None
        self.num_classes_ = None

    def _build_model(self, input_dim, output_dim):
        raise NotImplementedError

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        seed_device(self.device, self.random_state)

        self.num_classes_ = y.shape[1]
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        train_loader = DataLoader(
            AudioFeatureDataset(X_train, Y_train), batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            AudioFeatureDataset(X_val, Y_val), batch_size=self.batch_size, shuffle=False
        )

        self.model_ = self._build_model(X.shape[1], self.num_classes_).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        loss_fn = _SDTRLoss()
        use_amp = self.device.type == "cuda"
        if use_amp:
            scaler_amp = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for _ in range(self.epochs):
            self.model_.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = loss_fn(self.model_(x_batch), y_batch)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss = loss_fn(self.model_(x_batch), y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                    optimizer.step()

            self.model_.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    val_loss_accum += loss_fn(self.model_(x_batch), y_batch).item() * x_batch.size(
                        0
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
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(X_t)
            return out[0] if isinstance(out, tuple) else out

    def predict_proba(self, X):
        return torch.softmax(self._get_logits(X), dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return -y[np.arange(len(y)), preds].mean()


# ── Public estimators ──────────────────────────────────────────────────────


class SingleSDTR(_BaseSDTRWrapper):
    def __init__(
        self,
        num_trees=10,
        depth=6,
        lmbda=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
        epochs=100,
        patience=20,
        random_state=42,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            random_state=random_state,
        )
        self.num_trees = num_trees
        self.depth = depth
        self.lmbda = lmbda

    def _build_model(self, input_dim, output_dim):
        return _SDTR(
            in_features=input_dim,
            num_trees=self.num_trees,
            depth=self.depth,
            tree_dim=output_dim,
            flatten_output=True,
            lmbda=self.lmbda,
        )


class BoostedSDTR(_BaseSDTRWrapper):
    def __init__(
        self,
        num_boosting_layers=3,
        num_trees=10,
        depth=6,
        lmbda=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
        epochs=100,
        patience=20,
        random_state=42,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            random_state=random_state,
        )
        self.num_boosting_layers = num_boosting_layers
        self.num_trees = num_trees
        self.depth = depth
        self.lmbda = lmbda

    def _build_model(self, input_dim, output_dim):
        return _BoostedSDTRNet(
            in_features=input_dim,
            num_boosting_layers=self.num_boosting_layers,
            num_trees=self.num_trees,
            depth=self.depth,
            tree_dim=output_dim,
            flatten_output=True,
            lmbda=self.lmbda,
        )
