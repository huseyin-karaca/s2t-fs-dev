"""
FASTT — Feature-Adaptive Speech-to-Text Model Selection.

Two optimization paths:
  FASTTBoosted      — Differentiable path (Algorithm 2): Boosted SDTs with
                      per-round learnable feature transforms.
  FASTTAlternating  — Non-differentiable path (Algorithm 3): Alternating
                      optimization between a learnable transform and a
                      non-differentiable selector (e.g., XGBoost).
"""

from s2t_fs.models.fastt.fastt_alternating import FASTTAlternating
from s2t_fs.models.fastt.fastt_boosted import FASTTBoosted

__all__ = ["FASTTBoosted", "FASTTAlternating"]
