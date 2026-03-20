from s2t_fs.utils.logger import custom_logger as logger
import optuna
import logging

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

logger.info("Starting real optuna study test")
study = optuna.create_study()
study.optimize(objective, n_trials=3)
logger.info("Finished real optuna study test")
