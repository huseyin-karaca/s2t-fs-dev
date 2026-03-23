"""
Learnable feature transformations for the FASTT framework (Section III-C).

All transforms are nn.Module subclasses mapping z ∈ R^p → u ∈ R^d'.
Each exposes a `regularization_loss()` method returning the l1/l2 penalty
so that the training loop can add it to the expected-WER objective.
"""

import torch
import torch.nn as nn


class DiagonalGating(nn.Module):
    """Feature-wise scaling: u = q ⊙ z (Section III-C.1).

    Each component q_j assigns an importance weight to feature dimension j.
    Sparsity is encouraged via combined l1/l2 regularization on q.
    """

    def __init__(self, in_features: int, lambda1: float = 0.01, lambda2: float = 0.01):
        super().__init__()
        self.q = nn.Parameter(torch.ones(in_features))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.q * z

    def regularization_loss(self) -> torch.Tensor:
        return self.lambda1 * self.q.abs().sum() + self.lambda2 * (self.q ** 2).sum()


class LinearTransform(nn.Module):
    """Full linear projection: u = Wz + b (Section III-C.2).

    Allows reweighting and mixing of feature dimensions. Choose d' < p
    for dimensionality reduction.
    """

    def __init__(self, in_features: int, out_features: int, weight_decay: float = 1e-4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.weight_decay = weight_decay

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

    def regularization_loss(self) -> torch.Tensor:
        return self.weight_decay * (self.linear.weight ** 2).sum()


class LowRankTransform(nn.Module):
    """Low-rank factorization: u = W2 W1 z (Section III-C.3).

    W1 ∈ R^{r×p}, W2 ∈ R^{d'×r} with r ≪ min(p, d'). Constrains the rank
    of the projection while capturing inter-feature interactions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bottleneck_dim: int,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.w1 = nn.Linear(in_features, bottleneck_dim, bias=False)
        self.w2 = nn.Linear(bottleneck_dim, out_features, bias=False)
        self.weight_decay = weight_decay

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(z))

    def regularization_loss(self) -> torch.Tensor:
        return self.weight_decay * (
            (self.w1.weight ** 2).sum() + (self.w2.weight ** 2).sum()
        )


class NonlinearTransform(nn.Module):
    """Nonlinear bottleneck: u = W2 σ(W1 z) (Section III-C.4).

    σ is GELU activation. W1 ∈ R^{r×p}, W2 ∈ R^{d'×r} with bottleneck
    dimension r ≪ min(p, d').
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bottleneck_dim: int,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.w1 = nn.Linear(in_features, bottleneck_dim)
        self.activation = nn.GELU()
        self.w2 = nn.Linear(bottleneck_dim, out_features, bias=False)
        self.weight_decay = weight_decay

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(z)))

    def regularization_loss(self) -> torch.Tensor:
        return self.weight_decay * (
            (self.w1.weight ** 2).sum() + (self.w2.weight ** 2).sum()
        )


class IdentityTransform(nn.Module):
    """No-op transform (raw features baseline)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)


TRANSFORM_REGISTRY = {
    "identity": IdentityTransform,
    "diagonal": DiagonalGating,
    "linear": LinearTransform,
    "low_rank": LowRankTransform,
    "nonlinear": NonlinearTransform,
}


def build_transform(
    transform_type: str,
    in_features: int,
    out_features: int | None = None,
    bottleneck_dim: int | None = None,
    **kwargs,
) -> nn.Module:
    """Factory for feature transforms.

    Parameters
    ----------
    transform_type : str
        One of 'identity', 'diagonal', 'linear', 'low_rank', 'nonlinear'.
    in_features : int
        Input dimensionality p.
    out_features : int or None
        Output dimensionality d'. Defaults to in_features.
    bottleneck_dim : int or None
        Bottleneck rank r for low_rank and nonlinear transforms.
    **kwargs
        Passed to the transform constructor (e.g., lambda1, lambda2, weight_decay).
    """
    if transform_type not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform_type '{transform_type}'. "
            f"Choose from {list(TRANSFORM_REGISTRY.keys())}."
        )

    if out_features is None:
        out_features = in_features

    cls = TRANSFORM_REGISTRY[transform_type]

    if transform_type == "identity":
        return cls()
    elif transform_type == "diagonal":
        return cls(in_features=in_features, **kwargs)
    elif transform_type == "linear":
        return cls(in_features=in_features, out_features=out_features, **kwargs)
    elif transform_type in ("low_rank", "nonlinear"):
        if bottleneck_dim is None:
            bottleneck_dim = max(in_features // 4, 8)
        return cls(
            in_features=in_features,
            out_features=out_features,
            bottleneck_dim=bottleneck_dim,
            **kwargs,
        )
