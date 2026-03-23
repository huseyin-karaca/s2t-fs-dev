# s2t_fs/models/registry.py

import importlib

import optuna

from s2t_fs.utils.logger import custom_logger as logger


def instantiate_from_path(class_path: str, **kwargs):
    """
    String bir modül yolundan (örn: 's2t_fs.models.adastt_xgboost.AdaSTTXGBoost')
    sınıfı dinamik olarak import eder ve başlatır.
    """
    try:
        # Yolu modül ve sınıf olarak ikiye böl (sağdan ilk noktaya göre)
        module_path, class_name = class_path.rsplit(".", 1)

        # Modülü import et
        module = importlib.import_module(module_path)

        # Sınıfı modülün içinden çek
        cls = getattr(module, class_name)

        # Sınıfı başlat ve döndür (varsa init_args ile)
        return cls(**kwargs)

    except (ImportError, AttributeError) as e:
        logger.error(f"Dinamik yükleme başarısız oldu: {class_path}. Hata: {e}")
        raise


def build_optuna_space(space_config: dict):
    """
    Basit bir sözlüğü, Optuna dağılım nesnelerine çevirir.
    Şimdilik her şeyi Categorical kabul ediyoruz.
    """
    return {k: optuna.distributions.CategoricalDistribution(v) for k, v in space_config.items()}


def prepare_models_from_config(models_config: dict):
    """
    Config sözlüğündeki model tanımlarını okur, modelleri dinamik yükler
    ve uygun formata {name: (instance, space)} çevirir.
    """
    prepared_models = {}

    for name, m_cfg in models_config.items():
        class_path = m_cfg["class_path"]
        init_args = m_cfg.get("init_args", {})

        logger.debug(f"{name} modeli {class_path} üzerinden dinamik olarak yükleniyor...")

        # Helper'ları kullanarak modeli ve Optuna uzayını oluştur
        model_instance = instantiate_from_path(class_path, **init_args)
        optuna_space = build_optuna_space(m_cfg["hyperparameters"])

        prepared_models[name] = (model_instance, optuna_space)

    logger.debug(f"Toplam {len(prepared_models)} model deney için başarıyla hazırlandı.")
    return prepared_models


def prepare_model_from_config(model_params: dict):
    """
    Config sözlüğündeki tekil model tanımını okur, modeli dinamik yükler
    ve uygun formata (model_name, instance, space) çevirir.
    """

    model_name = model_params["model_name"]
    class_path = model_params["class_path"]
    init_args = model_params.get("init_args", {})

    logger.debug(f"{model_name} modeli {class_path} üzerinden dinamik olarak yükleniyor...")

    # Helper'ları kullanarak modeli ve Optuna uzayını oluştur
    model_instance = instantiate_from_path(class_path, **init_args)
    optuna_space = build_optuna_space(model_params["hyperparameters"])

    logger.debug(f"{model_name} modeli deney için başarıyla hazırlandı.")

    return model_name, model_instance, optuna_space


