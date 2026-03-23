# S2T-FS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Source codes and reproducibility package of the correspondng research paper

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         the Python package under src/ and configuration for tools like ruff
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project (importable as `src.*`).
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── data/                   <- Data loading / preparation
    ├── experiment/             <- Experiment runner(s)
    ├── models/                 <- Model implementations / registry
    └── utils/                  <- Small shared utilities
```

--------

To run experiments:

```bash 
python -m s2t_fs.experiment.synthetic.test-diagonal-transform --config configs/synt-exp_fastt-alternating-diagonal.json
python -m s2t_fs.experiment.synthetic.test-diagonal-transform --config configs/synt-exp_fastt-boosted-diagonal.json
```

or using mlflow run 

```bash
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test-diagonal-transform -P config=configs/synt-exp_fastt-alternating-diagonal.json
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test-diagonal-transform -P config=configs/synt-exp_fastt-boosted-diagonal.json
```


## A note ion MLflow tracking uri:

When you run an experiment natively using `python -m s2t_fs.experiment.synthetic.test-diagonal-transform ...`, your script reads the JSON config, sees the `tracking_uri`, and initiates connection to `sqlite:///s2t-fs-experiments.db`. This works perfectly.

However, when you use **`mlflow run`**, the `mlflow` CLI actually creates the run **before** your Python script is ever executed! 
Because `mlflow run` doesn't know about the [configs/synt-exp_fastt-alternating-diagonal.json](cci:7://file:///Users/huseyinkaraca/Desktop/s2t-fs/configs/synt-exp_fastt-alternating-diagonal.json:0:0-0:0) file natively, it defaults to creating the run in your local `./mlruns` folder. 

When your Python script finally starts, it sees that `mlflow run` already created an active run, and intentionally skips applying your JSON's `tracking_uri` to avoid breaking the context (which was the "Run not found" bug we fixed earlier).

### The Solution
If you want to use `mlflow run`, you must tell the `mlflow` CLI where the database is by exporting the environment variable **before** you execute the command. The configuration JSON's `tracking_uri` acts only as a fallback for running natively.

To make it work, run this in your terminal:
```bash
export MLFLOW_TRACKING_URI=sqlite:///s2t-fs-experiments.db
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test-diagonal-transform -P config=configs/synt-exp_fastt-alternating-diagonal.json
```

If you ever want the `tracking_uri` in the JSON to take charge without exporting environment variables, simply run the module directly without `mlflow run`:
```bash
python -m s2t_fs.experiment.synthetic.test-diagonal-transform --config configs/synt-exp_fastt-alternating-diagonal.json
``` 