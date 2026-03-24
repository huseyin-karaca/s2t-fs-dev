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
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test_transform -P config=configs/synt-exp_fastt-alternating-diagonal.json

# You can also use other non-diagonal transformations!
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test_transform -P config=configs/synt-exp_fastt-alternating-linear.json
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test_transform -P config=configs/synt-exp_fastt-boosted-lowrank.json
mlflow run . -e synthetic_experiment --env-manager=local -P script_name=test_transform -P config=configs/synt-exp_fastt-alternating-nonlinear.json
```

If you ever want the `tracking_uri` in the JSON to take charge without exporting environment variables, simply run the module directly without `mlflow run`:
```bash
python -m s2t_fs.experiment.synthetic.test_transform --config configs/synt-exp_fastt-alternating-diagonal.json
```  


# Coding Philosophy (as meta prompt)

**Project Architecture & General Philosophy:**
This is a strictly **Modular and Config-Driven Machine Learning Project**. You are acting as a Senior MLOps Engineer. Your primary goal is reusability, reproducibility, and architectural consistency. You must think globally about the project structure before writing any local code. 

**Rule 1: Strict Modularity & DRY (Don't Repeat Yourself)**
* **Never write from scratch if it exists:** If asked to write a test, experiment, or training script, do NOT build models or core logic from scratch in that file.
* **Import over implement:** You must import existing models, utilities, and components from their respective module folders.
* **Respect Directory Structure:** Any new experiment, script, or feature must be placed in its conceptually correct folder (e.g., experimental code goes strictly into `s2t_fs/experiment`).

**Rule 2: Standardized Interfaces (The Scikit-Learn Contract)**
* **Estimator API:** Every model implemented in this project MUST be wrapped or written as a fully compatible `scikit-learn` estimator. 
* **Hyperparameter Tuning:** Because of Rule 2, we exclusively use `Optuna` (specifically `OptunaSearchCV`) for hyperparameter tuning. 
* **Tuning as an Estimator:** Remember that an `OptunaSearchCV` object is an estimator itself. You should seamlessly use this object in pipelines and experiments without writing custom tuning loops.

**Rule 3: Centralized Tracking & Logging (MLOps Standard)**
* **MLflow is Mandatory:** We value rigorous experiment tracking. All runs, parameters, metrics, and models must be logged using MLflow.
* **Use Existing Loggers:** Use the project's existing clean, custom logger implementations. Do NOT write custom or print-based logging scripts for new files.
* **Refactor, Don't Bypass:** If you find the current logging style or MLflow integration inadequate or buggy for a specific task, **STOP**. Do not write a one-off workaround. Notify me and suggest an update to the core logging modules instead.

**Rule 4: Config-Driven Execution**
* **No Hardcoding:** Code execution must be treated as a mapping of "Config Inputs" $\rightarrow$ "Results Outputs".
* Experiments and models should be instantiated via configuration dictionaries/files.
* Both the input configurations and the output results/artifacts must be logged to MLflow to ensure 100% reproducibility.

**Agent Behavioral Directive:** When given a task, do not just fulfill the immediate request blindly. First, analyze how the solution fits into the `s2t_fs` modular structure, identify which existing modules can be reused, and ensure the final output adheres strictly to the Sklearn/Optuna/MLflow pipeline philosophy.