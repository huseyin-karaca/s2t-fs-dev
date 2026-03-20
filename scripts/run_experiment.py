# scripts/run_experiment.py
import argparse
import logging
import json
import time
import os
import mlflow

# Gördüğün gibi her şey s2t_fs altından geliyor. Script'in içi tertemiz.
from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.models.registry import prepare_model_registry
from s2t_fs.experiment.runner import run_nested_evaluation
from s2t_fs.utils.io import save_experiment_results

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # 1. Veri Hazırlığı (Bağımsız Modül)
    X_train, Y_train, X_test, Y_test, stats = load_and_prepare_data(cfg["data_params"])

    # 2. Modelleri Çek (Bağımsız Modül)
    models = prepare_model_registry(cfg["hyperparameter_spaces"])

    # MLflow Setup
    run_kwargs = {"run_id": os.environ.get("MLFLOW_RUN_ID")} if os.environ.get("MLFLOW_RUN_ID") else {"run_name": "InnerRun"}
    
    with mlflow.start_run(**run_kwargs):
        mlflow.log_params(cfg["data_params"])
        
        # 3. Asıl Deneyi Koş
        margin, wers, searches = run_nested_evaluation(
            models=models, 
            X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
            search_params=cfg["search_params"]
        )

        # 4. Sonuçları Kaydet
        mlflow.log_metric("margin", margin)
        for name, wer in wers.items():
            mlflow.log_metric(f"wer_{name}", wer)
            
        save_experiment_results(cfg, stats, wers, margin, searches)

if __name__ == "__main__":
    main()



# # scripts/run_experiment.py
# # ... importlar ...
# from s2t_fs.logger import custom_logger as logger

# def main():
#     # ... config yükleme ...
    
#     logger.info(f"Deney başlatılıyor. Config: {args.config}")
    
#     # 1. BÜYÜ: MLflow Level 1 (Top Level Log - Experimental Setting)
#     # Bu run, veri seti parametrelerini ve global ayarları tutar.
#     run_name = f"Exp_{cfg['data_params']['dataset']}_sub{cfg['data_params']['row_subsample']}"
    
#     with mlflow.start_run(run_name=run_name) as parent_run:
#         logger.info(f"MLflow Parent Run ID: {parent_run.info.run_id}")
        
#         # Sadece en üst seviyeyi ilgilendiren parametreleri (Data, Seed vb.) buraya logla
#         mlflow.log_params(cfg["data_params"])
#         mlflow.log_params({"global_seed": cfg["search_params"]["seed"]})
        
#         X_train, Y_train, X_test, Y_test, stats = load_and_prepare_data(cfg["data_params"])
#         models = prepare_model_registry(cfg["hyperparameter_spaces"])

#         # İç içe (nested) değerlendirmeyi çağır.
#         # Bu fonksiyon kendi içindeki Level 2 ve Level 3 loglamalarını halledecek.
#         margin, wers, searches = run_nested_evaluation(
#             models=models, 
#             X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,
#             search_params=cfg["search_params"]
#         )

#         # Tüm modeller bittikten sonra "En Üst Seviye" özet metrikleri ana run'a yaz
#         logger.info(f"Tüm modeller değerlendirildi. Kazanç marjı (Margin): %{margin:.2f}")
#         mlflow.log_metric("final_margin_vs_baseline", margin)
        
#         save_experiment_results(cfg, stats, wers, margin, searches)

# if __name__ == "__main__":
#     main()