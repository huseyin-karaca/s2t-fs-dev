# import optuna
# import logger
# import mlflow
# from sklearn.model_selection import ShuffleSplit
# from optuna.integration import OptunaSearchCV

# def run_comparative_experiment(model_registry, X_train, Y_train, X_test, Y_test, search_params):

#     seed = search_params["seed"]
#     n_trials = search_params["num_samples"]

#     cv = ShuffleSplit(
#         n_splits=search_params["num_folds"],
#         test_size=search_params["test_size"],
#         random_state=seed)

#     optuna.logging.set_verbosity(optuna.logging.WARNING)
#     mlflow.sklearn.autolog(log_models=False)

#     wers, searches = {}, {}
#     n = len(model_registry)
#     for i, (model_instance, config_dict, name) in enumerate(model_registry, 1):
#         logger.info("[%d/%d] %s — tuning (%d trials)...", i, n, name, n_trials)
#         search = OptunaSearchCV(
#             estimator=model_instance,
#             param_distributions=config_dict,
#             cv=cv,
#             n_trials=n_trials,
#             random_state=seed,
#             scoring=None,
#             refit=search_params["refit"],
#             return_train_score=True,
#         )

#         with mlflow.start_run(run_name=f"Inner_{name}", nested=True):
#             search.fit(X_train, Y_train)
#             searches[name] = search

#             preds = search.best_estimator_.predict(X_test)
#             wers[name] = float(Y_test[np.arange(len(Y_test)), preds].mean())
#             logger.info("  -> test WER: %.4f", wers[name])

#             # autolog handles best_params; we only log our custom business metric
#             mlflow.log_metric("test_wer", wers[name])
#             cv_res = search.cv_results_
#             best_idx = search.best_index_
#             mlflow.log_metric("cv_mean_test_score", -float(cv_res["mean_test_score"][best_idx]))

#     competitors = [v for k, v in wers.items() if not k.startswith("BoostedSDTR")]
#     baseline_min = min(competitors) if competitors else 1.0
#     main_wer = wers.get("BoostedSDTR", 0.0)
#     margin = ((main_wer - baseline_min) / baseline_min) * 100 if baseline_min > 0 and main_wer > 0 else 0.0

#     return margin, wers, searches


# s2t_fs/experiment/runner.py

import mlflow
import numpy as np
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import ShuffleSplit

from s2t_fs.utils.logger import custom_logger as logger


def run_nested_evaluation(models, X_train, Y_train, X_test, Y_test, search_params):

    seed = search_params["seed"]
    n_trials = search_params["num_samples"]

    cv = ShuffleSplit(
        n_splits=search_params["num_folds"],
        test_size=search_params["test_size"],
        random_state=seed,
    )

    # 1. BÜYÜ: Autolog'u sessizce aç. Sklearn/Optuna modellerini eğitirken
    # tüm hiperparametreleri ve CV skorlarını kendisi MLflow'a yazacak. (Level 3)
    mlflow.sklearn.autolog(log_models=False, silent=True)

    wers, searches = {}, {}

    for model_name, (model_instance, model_hpt_space) in models.items():
        # 2. BÜYÜ: Loguru'ya "Şu an hangi modeldeyiz" bilgisini bağla (Contextualize)
        # Böylece bu bloktaki tüm loglar otomatik olarak model adıyla etiketlenir.
        with logger.contextualize(model=model_name):
            logger.bind(category="Inner-Run").info(
                f"Hyperparameter tuning of {model_name}: ({n_trials} trials)"
            )

            search = OptunaSearchCV(
                estimator=model_instance,
                param_distributions=model_hpt_space,
                cv=cv,
                n_trials=n_trials,
                random_state=seed,
                scoring=None,
                refit=search_params["refit"],
                return_train_score=True,
            )

            # 3. BÜYÜ: MLflow Level 2 (Second Level Log - Best of Model)
            # nested=True sayesinde bu run, ana scriptteki (Level 1) run'ın altına girer.
            with mlflow.start_run(run_name=f"Model_Tuning_{model_name}", nested=True):
                # Fit çalıştığı anda, Sklearn autolog devreye girer.
                # OptunaSearchCV'nin denediği HER BİR parametre kombinasyonunu (Level 3)
                # MLflow'a bu "Model_Tuning" run'ının altına "child" olarak sessizce kaydeder.
                search.fit(X_train, Y_train)
                searches[model_name] = search

                # En iyi tahminleri yap ve iş metriklerimizi (Business Metrics) hesapla
                preds = search.best_estimator_.predict(X_test)
                wers[model_name] = float(Y_test[np.arange(len(Y_test)), preds].mean())

                logger.bind(category="HPT-Detail").success(
                    f"Tuning of {model_name} has been completed. Best Test WER: {wers[model_name]:.4f}"
                )

                # Kendi özel metriklerimizi manuel olarak logluyoruz (Level 2'ye)
                mlflow.log_metric("custom_test_wer", wers[model_name])

                best_idx = search.best_index_
                cv_res = search.cv_results_
                mlflow.log_metric(
                    "cv_mean_test_score", -float(cv_res["mean_test_score"][best_idx])
                )

    # --- Baseline ve Margin Hesaplamaları ---
    competitors = [v for k, v in wers.items() if not k.startswith("BoostedSDTR")]
    baseline_min = min(competitors) if competitors else 1.0
    main_wer = wers.get("BoostedSDTR", 0.0)
    margin = (
        ((main_wer - baseline_min) / baseline_min) * 100
        if baseline_min > 0 and main_wer > 0
        else 0.0
    )

    return margin, wers, searches
