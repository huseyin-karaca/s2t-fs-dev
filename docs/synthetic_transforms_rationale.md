# Rationale regarding Alternate Datasets for Non-Diagonal Transform Experiments

When exploring and verifying the `FASTT` framework, specifically when transitioning our test suite from `diagonal` transforms to matrix-based projection transforms (`linear`, `low_rank`, and `nonlinear`), it is important to test against variations in dataset complexities.

**1. Linear Transform (`Configs: synt-exp_fastt-alternating-linear`)**
- Data Configuration Shift: `n_samples=4500`, `n_informative=6`, `n_noise=30`
- **Reasoning:** A dense linear projection effectively mixes all feature dimensions. Providing slightly more informative features amidst a larger set of noise allows us to verify if the matrix factorization correctly isolates important vectors compared to scalar elements.

**2. Low Rank Transform (`Configs: synt-exp_fastt-boosted-lowrank`)**
- Data Configuration Shift: `n_samples=5000`, `n_noise=45`, `n_experts=4`
- **Reasoning:** A `low_rank` transformation enforces an informational bottleneck. Introducing highly excessive noise and more redundant experts tests whether the model acts efficiently as a strong global feature reduction system while keeping the parameter footprint smaller than full linear layers.

**3. Nonlinear Transform (`Configs: synt-exp_fastt-alternating-nonlinear`)**
- Data Configuration Shift: `n_samples=3000`, `n_noise=10`
- **Reasoning:** The `nonlinear` matrix transform builds complex interrelations via `GELU` activations. While it's powerful, we keep the dataset simpler to ensure convergence behaviors inside the surrogate training pipeline (`Alternating`) can map complex activation outputs without suffering from optimization vanishing/exploding bounds.

All output models successfully derive global importance using $L_2$ column approximations from their input weights, ensuring compatibility.
