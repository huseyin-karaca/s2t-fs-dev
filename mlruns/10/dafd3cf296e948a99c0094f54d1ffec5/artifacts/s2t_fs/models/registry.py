# s2t_fs/models/registry.py

import importlib

import optuna

from s2t_fs.utils.logger import custom_logger as logger


def instantiate_from_path(class_path: str, **kwargs):
    """Dynamically import and instantiate a class from a dotted path.

    Example: instantiate_from_path('s2t_fs.models.adastt_xgboost.AdaSTTXGBoost')
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)
    except (ImportError, AttributeError) as e:
        logger.error(f"Dynamic import failed for {class_path}: {e}")
        raise


def resolve_nested_configs(init_args):
    """Recursively resolve nested estimator configs in init_args or lists.

    If a value is a dict containing 'class_path', it is treated as an
    estimator spec and instantiated via instantiate_from_path.
    If a value is a list (e.g., an sklearn Pipeline's steps), it recursively attempts to resolve dicts inside it.
    This enables config-driven composition (e.g., FASTTAlternating's base_selector or Pipeline steps).
    """
    if isinstance(init_args, dict):
        if "class_path" in init_args:
            nested_args = resolve_nested_configs(init_args.get("init_args", {}))
            resolved_cls = instantiate_from_path(init_args["class_path"], **nested_args)
            logger.debug(f"Nested estimator resolved from {init_args['class_path']}")
            return resolved_cls
        else:
            resolved = {}
            for key, val in init_args.items():
                resolved[key] = resolve_nested_configs(val)
            return resolved
    elif isinstance(init_args, list):
        return [resolve_nested_configs(item) for item in init_args]
    else:
        return init_args


def build_optuna_space(space_config: dict):
    """
    Basit bir sözlüğü, Optuna dağılım nesnelerine çevirir.
    Şimdilik her şeyi Categorical kabul ediyoruz.
    """
    return {k: optuna.distributions.CategoricalDistribution(v) for k, v in space_config.items()}


def prepare_models_from_config(models_config: dict):
    """Read model definitions from config, dynamically load and return them.

    Returns {name: (instance, optuna_space)}.
    """
    prepared_models = {}

    for name, m_cfg in models_config.items():
        class_path = m_cfg["class_path"]
        init_args = resolve_nested_configs(m_cfg.get("init_args", {}))

        logger.debug(f"Loading model '{name}' from {class_path}...")

        model_instance = instantiate_from_path(class_path, **init_args)
        optuna_space = build_optuna_space(m_cfg["hyperparameters"])

        prepared_models[name] = (model_instance, optuna_space)

    logger.debug(f"{len(prepared_models)} models prepared for experiment.")
    return prepared_models


def prepare_model_from_config(model_params: dict):
    """Read a single model definition from config, load and return it.

    Returns (model_name, instance, optuna_space).
    """
    model_name = model_params["model_name"]
    class_path = model_params["class_path"]
    init_args = resolve_nested_configs(model_params.get("init_args", {}))

    logger.debug(f"Loading model '{model_name}' from {class_path}...")

    model_instance = instantiate_from_path(class_path, **init_args)
    optuna_space = build_optuna_space(model_params["hyperparameters"])

    logger.debug(f"Model '{model_name}' ready.")

    return model_name, model_instance, optuna_space


def instantiate_model_from_config(model_cfg: dict):
    """Instantiate a single model from a config dict (without HPT space).

    Simpler variant of prepare_model_from_config for experiments that
    don't use OptunaSearchCV (e.g., synthetic validation).
    """
    class_path = model_cfg["class_path"]
    init_args = resolve_nested_configs(model_cfg.get("init_args", {}))
    return instantiate_from_path(class_path, **init_args)


