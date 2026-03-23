"""
FASTT-XGBoost: Non-differentiable path (Algorithm 3).

Alternating optimization between a learnable feature transform and
a non-differentiable selector (e.g., XGBoost with custom WER objective).

At each alternating iteration t:
  1. Transform features:  u = Tθ(z)
  2. Selector update:     Train selector fψ on {u_i} with custom WER targets
  3. Transform update:    Fix fψ, optimize θ via surrogate gradient

The surrogate gradient is obtained by fitting a linear model that
approximates the selector's input→logit mapping, enabling backpropagation
through the transform.
"""

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from s2t_fs.models.fastt.transforms import build_transform


class _LinearSurrogate(nn.Module):
    """Differentiable linear approximation of the selector's mapping u → q.

    Fitted to match the non-differentiable selector's predictions on training
    data, then used to propagate gradients back through the transform.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.linear(u)

    def fit(self, u_np: np.ndarray, q_np: np.ndarray, lr=1e-2, steps=200):
        """Least-squares fit of surrogate to selector outputs."""
        u_t = torch.tensor(u_np, dtype=torch.float32)
        q_t = torch.tensor(q_np, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.linear(u_t), q_t)
            loss.backward()
            optimizer.step()
        return self


class FASTTAlternating(BaseEstimator, ClassifierMixin):
    """FASTT for non-differentiable selectors (Algorithm 3).

    Jointly learns a feature transform and a non-differentiable selector
    (e.g., XGBoost) via alternating optimization with surrogate gradients.

    Parameters
    ----------
    base_selector : BaseEstimator
        A selector with custom WER objective. Must implement fit(X, y),
        predict(X), and predict_proba(X). E.g., AdaSTTXGBoost.
    transform_type : str
        Feature transform variant: 'diagonal', 'linear', 'low_rank', 'nonlinear'.
    transform_kwargs : dict or None
        Extra arguments for the transform (e.g., lambda1, bottleneck_dim).
    num_iterations : int
        Number of alternating optimization iterations T.
    transform_lr : float
        Learning rate for the transform update step.
    transform_steps : int
        Gradient steps per transform update.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        base_selector=None,
        transform_type="diagonal",
        transform_kwargs=None,
        num_iterations=5,
        transform_lr=1e-3,
        transform_steps=100,
        random_state=42,
    ):
        self.base_selector = base_selector
        self.transform_type = transform_type
        self.transform_kwargs = transform_kwargs
        self.num_iterations = num_iterations
        self.transform_lr = transform_lr
        self.transform_steps = transform_steps
        self.random_state = random_state

        self.transform_ = None
        self.selector_ = None
        self.num_classes_ = None

    def _get_selector_logits(self, X):
        """Get raw logits from the selector (before softmax)."""
        probs = self.selector_.predict_proba(X)
        logits = np.log(np.clip(probs, 1e-10, None))
        return logits

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.num_classes_ = y.shape[1]
        in_features = X.shape[1]

        t_kwargs = self.transform_kwargs or {}
        self.transform_ = build_transform(
            self.transform_type, in_features=in_features, **t_kwargs
        )

        z_tensor = torch.tensor(X, dtype=torch.float32)

        for iteration in range(self.num_iterations):
            # ── Step 1: Transform features ───────────────────────────────
            with torch.no_grad():
                u_tensor = self.transform_(z_tensor)
            u_np = u_tensor.numpy().astype(np.float32)

            # ── Step 2: Selector update (fix θ, optimize ψ) ─────────────
            self.selector_ = clone(self.base_selector)
            self.selector_.fit(u_np, y)

            # ── Step 3: Transform update (fix ψ, optimize θ) ────────────
            # 3a. Get selector's predictions as logits
            q_np = self._get_selector_logits(u_np)

            # 3b. Fit differentiable surrogate: linear model u → q
            u_dim = u_np.shape[1]
            surrogate = _LinearSurrogate(u_dim, self.num_classes_)
            self._fit_surrogate(surrogate, u_np, q_np)

            # 3c. Backprop through: z → Tθ(z) → surrogate → softmax → E[WER]
            wer_tensor = torch.tensor(y, dtype=torch.float32)
            self._update_transform(
                z_tensor, wer_tensor, surrogate, self.transform_lr, self.transform_steps
            )

        # Final selector fit on final transformed features
        with torch.no_grad():
            u_final = self.transform_(z_tensor).numpy().astype(np.float32)
        self.selector_ = clone(self.base_selector)
        self.selector_.fit(u_final, y)

        return self

    def _fit_surrogate(self, surrogate, u_np, q_np, lr=1e-2, steps=200):
        """Least-squares fit of linear surrogate to selector's logits."""
        u_t = torch.tensor(u_np, dtype=torch.float32)
        q_t = torch.tensor(q_np, dtype=torch.float32)
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(surrogate(u_t), q_t)
            loss.backward()
            optimizer.step()

    def _update_transform(self, z_tensor, wer_tensor, surrogate, lr, steps):
        """Update transform parameters via expected-WER + surrogate backprop."""
        transform_params = list(self.transform_.parameters())
        if not transform_params:
            return

        for param in surrogate.parameters():
            param.requires_grad_(False)

        optimizer = torch.optim.Adam(transform_params, lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            u = self.transform_(z_tensor)
            q_approx = surrogate(u)
            probs = torch.softmax(q_approx, dim=1)
            expected_wer = (probs * wer_tensor).sum(dim=1).mean()
            reg = self.transform_.regularization_loss()
            loss = expected_wer + reg
            loss.backward()
            nn.utils.clip_grad_norm_(transform_params, 1.0)
            optimizer.step()

    def predict_proba(self, X):
        z_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            u = self.transform_(z_tensor).numpy().astype(np.float32)
        return self.selector_.predict_proba(u)

    def predict(self, X):
        z_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            u = self.transform_(z_tensor).numpy().astype(np.float32)
        return self.selector_.predict(u)

    def score(self, X, y):
        """Negative mean WER (higher is better, sklearn convention)."""
        preds = self.predict(X)
        return -y[np.arange(len(y)), preds].mean()

    def get_gating_weights(self):
        """Return gating vector (only for diagonal transform)."""
        if self.transform_type != "diagonal" or self.transform_ is None:
            return None
        return self.transform_.q.detach().cpu().numpy()
