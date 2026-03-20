import sys
from pathlib import Path

# Proje ana dizinini sys.path'e ekle ki s2t_fs import edilebilsin
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from s2t_fs.utils.logger import custom_logger as logger
import logging

# -----------------------------------------------------------------------------------
def _log_completed_trial():
    # Optuna'nın loglama mekanizmasını taklit ediyoruz. 
    # Optuna, iç mekanizmasında `optuna.study.study` altında bir logger oluşturur.
    # Test ortamında Patcher mantığını (catch & tag) net görebilmek için
    # loguru'nun extra dictionary'sine doğrudan original_name enjekte ediyoruz.
    # (Gerçek kullanımda bunu InterceptHandler otomatik yapar)
    logger.bind(original_name="optuna.study.study").info(
        "Trial 0 finished with value: -0.1944. Best is trial 0 with value: -0.1944."
    )

# -----------------------------------------------------------------------------------
# MLflow Mock Function
# MLflow loglarını Patcher yakalayacak ve MLFLOW kategorisi basacak.
# -----------------------------------------------------------------------------------
def some_mlflow_process():
    mlflow_logger = logging.getLogger("mlflow.tracking")
    mlflow_logger.info("MLflow Parent Run başlatıldı. ID: 1234abcd")
    mlflow_logger.warning("DB bağlantısı biraz yavaş gerçekleşti.")

# -----------------------------------------------------------------------------------
# Doğrudan Loguru Kullanımı (Kategori ekleyerek)
# -----------------------------------------------------------------------------------
def data_process():
    # Artık logger.bind(category="DATA") kullanarak o mesaja kalıcı etiket veriyoruz.
    logger.bind(category="DATA").info("Veriler yüklendi. (1500 satır, 45 özellik)")
    logger.bind(category="DATA").success("Missed values imputation tamamlandı.")

def main():
    logger.info("=== Log Test Scripti Başlatıldı ===")
    
    # 1. Standart Kategori (Hiyerarşi testi)
    logger.bind(category="Inner-Run").info("Hyperparameter tuning of XGBoost... (50 trials)")
    
    # 2. Optuna Mock Testi
    _log_completed_trial()
    
    # 3. MLflow Mock Testi
    some_mlflow_process()
    
    # 4. Direkt çağrımlar
    data_process()
    
    # 4. Standart seviyeler (Kategorisi olmak zorunda değil!)
    logger.success("Tüm test akışları hatasız tamamlandı!")
    logger.warning("Bu bir uyarı testidir, kategorisiz basılabilir.")
    logger.error("Birşeyler ters gitti!")

if __name__ == "__main__":
    main()
