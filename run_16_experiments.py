import os
import json
import subprocess

CONFIG_DIR = "configs/16_exp"
os.makedirs(CONFIG_DIR, exist_ok=True)

transforms = ["diagonal", "linear", "low_rank", "nonlinear"]

# 1. FASTT Alternating Logic
base_models_alt = {
    "XGBoost": {
        "class_path": "s2t_fs.models.adastt_xgboost.AdaSTTXGBoost",
        "init_args": {"n_estimators": 50, "learning_rate": 0.1}
    },
    "LightGBM": {
        "class_path": "s2t_fs.models.adastt_lightgbm.AdaSTTLightGBM",
        "init_args": {"n_estimators": 50, "learning_rate": 0.1}
    }
}

for t in transforms:
    for name, b_cfg in base_models_alt.items():
        fname = f"{CONFIG_DIR}/exp_alt_{name}_{t}.json"
        
        cfg = {
            "mlflow_cfg": {
                "tracking_uri": "sqlite:///s2t-fs-experiments.db",
                "experiment_name": "Check",
                "run_name": f"FASTT-Alternating-{name}-{t.capitalize()}"
            },
            "data_cfg": {
                "n_samples": 1500,
                "n_informative": 4,
                "n_noise": 16,
                "n_experts": 2,
                "test_size": 0.2,
                "seed": 42
            },
            "search_cfg": {
                "seed": 42,
                "num_samples": 2,
                "num_folds": 1,
                "test_size": 0.1,
                "refit": True
            },
            "model_cfg": {
                "model_name": "FASTTAlternating",
                "class_path": "s2t_fs.models.fastt.fastt_alternating.FASTTAlternating",
                "init_args": {
                    "transform_type": t,
                    "transform_steps": 50,
                    "base_selector": b_cfg
                },
                "hyperparameters": {
                    "num_iterations": [2, 3],
                    "transform_lr": [0.01]
                }
            }
        }
        
        if t in ["low_rank", "nonlinear"]:
            cfg["model_cfg"]["init_args"]["transform_kwargs"] = {"bottleneck_dim": 4}
            
        with open(fname, "w") as f:
            json.dump(cfg, f, indent=2)

# 2. FASTT Boosted Logic
base_models_boost = {
    "SDTR": {
        "class_path_": "s2t_fs.models.sdtr_models._SDTR",
        "init_args_": {"num_trees": 5, "depth": 3, "lmbda": 0.1, "lmbda2": 0.01}
    },
    "MLP": {
        "class_path_": "s2t_fs.models.adastt_mlp._MLPDifferentiable",
        "init_args_": {"dropout_rate": 0.1}
    }
}

for t in transforms:
    for name, b_cfg in base_models_boost.items():
        fname = f"{CONFIG_DIR}/exp_boosted_{name}_{t}.json"
        
        cfg = {
            "mlflow_cfg": {
                "tracking_uri": "sqlite:///s2t-fs-experiments.db",
                "experiment_name": "Check",
                "run_name": f"FASTT-Boosted-{name}-{t.capitalize()}"
            },
            "data_cfg": {
                "n_samples": 1500,
                "n_informative": 4,
                "n_noise": 16,
                "n_experts": 2,
                "test_size": 0.2,
                "seed": 42
            },
            "search_cfg": {
                "seed": 42,
                "num_samples": 2,
                "num_folds": 1,
                "test_size": 0.1,
                "refit": True
            },
            "model_cfg": {
                "model_name": "FASTTBoosted",
                "class_path": "s2t_fs.models.fastt.fastt_boosted.FASTTBoosted",
                "init_args": {
                    "transform_type": t,
                    "num_rounds": 2,
                    "epochs": 5,
                    "base_estimator": b_cfg
                },
                "hyperparameters": {
                    "lr": [0.01, 0.005]
                }
            }
        }
        
        if t in ["low_rank", "nonlinear"]:
            cfg["model_cfg"]["init_args"]["transform_kwargs"] = {"bottleneck_dim": 4}
            
        with open(fname, "w") as f:
            json.dump(cfg, f, indent=2)


import glob
all_configs = glob.glob(f"{CONFIG_DIR}/*.json")

env = os.environ.copy()
env["MLFLOW_TRACKING_URI"] = "sqlite:///s2t-fs-experiments.db"
env["MLFLOW_EXPERIMENT_NAME"] = "Check"

for conf in all_configs:
    print(f"Running experiment for config: {conf}")
    cmd = [
        "mlflow", "run", ".", "-e", "synthetic_experiment",
        "--env-manager=local", "-P", "script_name=test_transform",
        "-P", f"config={conf}"
    ]
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"Experiment failed for {conf}")
        # Stop on failure for safety
        break
