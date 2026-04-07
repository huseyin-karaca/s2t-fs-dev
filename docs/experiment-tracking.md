# Experiment Tracking (MLflow)

All S2T-FS experiments are tracked with **MLflow**, using a local SQLite database as the
backend store. This page explains the setup, what is logged, and how to inspect results.

---

## Setup

Experiments are logged **locally** to `s2t-fs-experiments.db` in the project root.
No MLflow server is required. No environment variables need to be set — the tracking URI is
read from the config:

```json
"mlflow_params": {
  "tracking_uri": "sqlite:///s2t-fs-experiments.db",
  "experiment_name": "Model Comparison (VoxPopuli)"
}
```

!!! warning "Activate the Conda environment before any MLflow interaction"
    Always run `conda activate s2t-fs` before starting an experiment or launching the
    MLflow UI. Running MLflow outside this environment will cause severe version mismatch
    errors between MLflow, scikit-learn, and Optuna — these errors can be silent or
    produce misleading messages.

    ```bash
    conda activate s2t-fs
    mlflow ui --backend-store-uri sqlite:///s2t-fs-experiments.db
    ```

---

## Viewing Results

After running any experiment, launch the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///s2t-fs-experiments.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

If you are working on a remote machine (e.g., a Vast.ai instance), download the database
file first and then run the UI locally:

```bash
# On your local machine:
scp -P <port> root@<host>:/workspace/s2t-fs/s2t-fs-experiments.db .
mlflow ui --backend-store-uri sqlite:///s2t-fs-experiments.db
```

---

## What Gets Logged

Every run — at every tier — automatically records the following.

### Parameters

All config keys are flattened (e.g., `data_params.dataset`, `search_params.num_samples`) and
logged as MLflow parameters via `log_experiment_metadata()`. The full config JSON is also
uploaded as an artifact so every run is 100% reproducible from its logged state alone.

### Metrics

| Metric | Description |
|--------|-------------|
| `cv_mean_train_wer` | Mean cross-validation WER on the training folds |
| `cv_mean_test_wer` | Mean cross-validation WER on the validation fold |
| `custom_test_wer` | WER of the best estimator on the held-out test set |
| `our_best_wer` | Best WER among the target model(s) (Tier 2 / 3 parent runs) |
| `best_competitor_wer` | Best WER among all competitor models (Tier 2 / 3 parent runs) |
| `wer_margin` | `best_competitor_wer − our_best_wer` (positive = we win) |
| `best_margin` | Best margin found across all Optuna trials (Tier 3 parent run only) |

### Artifacts

| Artifact | Description |
|----------|-------------|
| `<config_name>.json` | The exact config file used for this run |
| `s2t_fs/` | A snapshot of the entire `s2t_fs/` source directory |
| `results/results.json` | CV and test metrics as a structured JSON file |
| `study_summary.json` | Full Optuna trial history (Tier 3 runs only) |
| `margin_summary.json` | Per-model WER results and margin details (Tier 2 / 3 parent runs) |
| Feature importance plots | Logged by the `plot_feature_importances` callback (when configured) |

### Tags

| Tag | Description |
|-----|-------------|
| `mlflow.source.git.commit` | Git commit hash at experiment time |
| `mlflow.parentRunId` | Set on all child runs to enable nested run navigation in the UI |

---

## Run Hierarchy in the UI

The MLflow UI displays nested runs. Navigate to any parent run and expand it to see its
children:

```
Tier 3: Margin_Optimization_Study
  └── Trial_0_Comparison
        ├── Model_Tuning_Dummy_0
        ├── Model_Tuning_Raw_XGBoost
        ├── Model_Tuning_FASTT_Boosted
        └── Model_Tuning_FASTT_Alternating
```

Each child run contains the full HPT trace for that model including all Optuna trial results.

---

## The `mlflow run` Workflow

Using `mlflow run .` adds a layer of MLflow orchestration that changes how the tracking URI
is resolved. Read this carefully before using `mlflow run`.

When you run `python -m s2t_fs.experiment --config ...`, the script reads the `tracking_uri`
from the JSON config and sets it directly. This always works.

When you run `mlflow run .`, the MLflow CLI **creates the parent run before your script
starts**, using whatever tracking URI is active in the shell environment. Your script detects
the pre-existing active run (`MLFLOW_RUN_ID` environment variable) and intentionally skips
overriding the tracking URI to avoid breaking the active run context.

**Consequence:** if you do not export `MLFLOW_TRACKING_URI` before calling `mlflow run`,
your results will be written to `./mlruns` (the default), not to `s2t-fs-experiments.db`.

**The correct pattern:**

```bash
MLFLOW_TRACKING_URI="sqlite:///s2t-fs-experiments.db" mlflow run . \
  -e model_comparison --env-manager=local \
  -P config=configs/model_comparison_voxpopuli.json \
  --experiment-name "Model Comparison (VoxPopuli)"
```

---

## Reproducibility Guarantee

Because every run logs its full config, source snapshot, and git commit hash, any completed
run can be reproduced exactly:

1. Check out the commit hash stored in `mlflow.source.git.commit`.
2. Download the config artifact from the run.
3. Run: `make run CONFIG=<downloaded_config.json>`

The logged source snapshot also serves as a backup if the repository has changed since the
run was executed.
