import mlflow
import argparse
import json

from s2t_fs.utils.logger import custom_logger as logger
from s2t_fs.data.loader import load_and_prepare_data
from s2t_fs.experiment.train_single_model import hpt_single_model
from s2t_fs.utils.mlflow_utils import log_experiment_metadata

def multi_model_experiment(cfg, config_path=None):

    mlflow_params = cfg["mlflow_params"]
    data_params = cfg["data_params"]

    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    # 2. Pre-load data ONCE
    logger.bind(category="Data").info("Pre-loading data for multi-model experiment...")
    preloaded_data = load_and_prepare_data(data_params)
    logger.bind(category="Data").success("Data loaded successfully.")

    results = {}

    target_model_name = cfg["evaluation_params"]["target_model_name"]
    models_dict = cfg["models"]

    # 3. Start Parent Run
    with mlflow.start_run(run_name="Multi_Model_Benchmark") as parent_run:
        
        # Log parent metadata
        log_experiment_metadata(cfg, config_path)
        
        # 4. Iterate over models and run HPT
        for model_name, model_cfg in models_dict.items():
            logger.bind(category="Orchestrator").info(f"Starting HPT for {model_name}...")
            
            # Create a temporary config for the single run
            single_cfg = {
                "mlflow_params": mlflow_params,
                "data_params": data_params,
                "search_params": cfg["search_params"],
                "models_config": {
                    model_name: model_cfg
                }
            }
            
            # Call single model tuning with preloaded data
            # nested=True inside hpt_single_model will make it a child of the current parent run
            best_wer = hpt_single_model(single_cfg, config_path=config_path, preloaded_data=preloaded_data)
            results[model_name] = best_wer
            
            logger.bind(category="Orchestrator").info(f"Finished HPT for {model_name}. Best WER: {best_wer:.4f}")

        # 5. Calculate Margins
        if target_model_name not in results:
            logger.error(f"Target model '{target_model_name}' not found in results!")
            return results

        our_wer = results[target_model_name]
        
        competitors = [wer for name, wer in results.items() if name != target_model_name]
        if competitors:
            best_competitor_wer = min(competitors)
            margin = best_competitor_wer - our_wer
            
            # Log metrics to parent run
            mlflow.log_metric("wer_margin", margin)
            mlflow.log_metric("best_competitor_wer", best_competitor_wer)
            mlflow.log_metric("our_wer", our_wer)
            
            logger.bind(category="Margin").success(
                f"Evaluation Completed. Our WER: {our_wer:.4f}, Best Competitor WER: {best_competitor_wer:.4f}, Margin: {margin:.4f}"
            )
        else:
            logger.warning("No competitors found to calculate margin.")

    return results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Multi-Model Hyperparameter Tuning Benchmark with MLFlow")

    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/multi_model_hpt_config.json", 
        help="Path to the JSON configuration file"
    )
    
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Start multi-model benchmark
    multi_model_experiment(cfg, config_path=args.config)
