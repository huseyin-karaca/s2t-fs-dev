# Context: S2T-FS (Speech-to-Text Feature Selection) Project

Hello! You are going to act as a Senior Machine Learning Engineer assisting me with the S2T-FS repository. Before we write any code, please carefully read and internalize our project description, design philosophy, and future roadmap. 

## 1. Brief Description
This is a machine learning experiment pipeline built in Python. The core objective is to tune, evaluate, and benchmark various models (like XGBoost, MLPs, Random Ensembles, and custom AdaSTT variants) for Speech-to-Text feature selection and evaluation tasks. 

Key technologies:
- **MLflow:** Used for comprehensive experiment tracking and logging.
- **Optuna:** Used for Hyperparameter Tuning (HPT) via `OptunaSearchCV`.
- **Scikit-learn / XGBoost:** Core modeling libraries.

Our main evaluation metric is WER (Word Error Rate). We often benchmark our proposed models against baseline competitors and calculate a "WER Margin" to prove superiority.

## 2. Design Philosophy
Your future code suggestions MUST adhere to these architectural principles:
- **Configuration-Driven:** The entire pipeline behavior is dictated by JSON configuration files. Configurations are strictly split into `data_params`, `search_params`, and flattened `model_params`.
- **Hierarchical MLflow Tracking:** We strictly use MLflow's Parent-Child run architecture (`nested=True`). An orchestrator script opens a Parent Run ("Multi-Model Benchmark") and logs cross-model metrics (like WER margins). Individual models and their subsequent Optuna HPT trials are automatically logged as Child Runs under this parent. In the future, we may further extend these parent-child relationships, such as dataset-wise searches (with different data loading / experimenting parameters) as grand-parent runs or including each hyperparameter trial as a grand-child run. 

---

**Your Directive:**
Please confirm that you have read and understood this architecture and its underlying philosophy. I would also appreciate your feedback. You are not limited to the comments above, these are not strict commands, but a brief sketch to inform you. If you want to drop any of them, or improve them, please let me know. I wait your feedbacks regarding the overall codes and my design philosophy. (I am a ML graduate student.)

I understand that there is no such thing as “perfect code,” but I aim to write clean, well-structured, and professional code. I believe that keeping the kitchen clean enhances both the cooking experience and the quality of the food—similarly, maintaining clean code improves both the development process and the final product.