# S2T-FS

**Feature Selection for Speech-to-Text Model Routing**

[![CCDS](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

---

## What Is This Project?

S2T-FS is the reproducibility package for a research paper on **adaptive feature selection for Speech-to-Text (S2T) model routing**. Given an audio utterance represented as an acoustic feature vector, the task is to predict which S2T model (among a pool of candidate models) will produce the lowest Word Error Rate (WER) for that utterance.

Rather than running all candidate models and selecting the best output post-hoc, S2T-FS learns a **lightweight, feature-selective routing policy** that makes this decision at inference time — with no additional transcription cost.

### Key Contributions

- **FASTT** (Feature-Adaptive Selector with Trainable Transforms): A novel framework that jointly learns a parametric feature transformation and a non-differentiable tree-based selector via alternating optimization.
- **Two FASTT variants**: `FASTTAlternating` for non-differentiable selectors (e.g., XGBoost) and `FASTTBoosted` for differentiable sequential layers.
- **Four transform families**: Diagonal gating, linear projection, low-rank bottleneck, and nonlinear (GELU) projection.
- **A 3-tier benchmarking pipeline**: Single-model HPT → multi-model comparison → competitive margin maximization (all tracked end-to-end in MLflow).
- **Full reproducibility**: Every experiment is config-driven, every result is logged with its full config, git commit hash, and source snapshot.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/huseyin-karaca/s2t-fs-dev.git
cd s2t-fs-dev

# 2. Create and activate the Conda environment
make create_environment
conda activate s2t-fs

# 3. Download the pre-processed datasets
make download_processed_data

# 4. Run the full multi-model benchmark on VoxPopuli
make run-comparison CONFIG=configs/model_comparison_voxpopuli.json
```

See [Getting Started](getting-started.md) for a complete installation guide.

---

## Datasets

All datasets are distributed as pre-processed Parquet files. Each file contains:

- `uid`: unique utterance identifier
- `f0, f1, ..., fN`: acoustic feature columns
- `wer_<model_name>`: per-model WER target columns

| Dataset | Description |
|---------|-------------|
| **VoxPopuli** | European Parliament speech recordings |
| **LibriSpeech** | Audiobook read speech |
| **AMI** | Meeting room recordings |
| **Common Voice** | Crowd-sourced speech |

---

## Navigation

| Section | What You Will Find |
|---------|--------------------|
| [Getting Started](getting-started.md) | Environment setup, data download, and first experiment |
| [Architecture](architecture.md) | Module layout, design philosophy, and the model registry |
| [Model Catalog](models.md) | Every model in the library with parameters and usage notes |
| [Experimental Design](experiments.md) | The 3-tier experiment hierarchy and config schema |
| [Orchestration & Parallelism](experiment_orchestration.md) | Execution modes, hardware guards, and speed guidance |
| [Experiment Tracking](experiment-tracking.md) | MLflow setup, viewing logs, and what gets recorded |
| [Remote Execution](remote-execution.md) | Renting a Vast.ai GPU and running experiments remotely |

---

## How to Cite

If you use this code or the accompanying paper in your work, please cite:

```bibtex
@article{karaca2025s2tfs,
  title   = {Feature-Adaptive Speech-to-Text Model Routing via Trainable Transforms},
  author  = {Karaca, H{\"u}seyin},
  year    = {2025}
}
```

---

## License

See `LICENSE` in the repository root.
