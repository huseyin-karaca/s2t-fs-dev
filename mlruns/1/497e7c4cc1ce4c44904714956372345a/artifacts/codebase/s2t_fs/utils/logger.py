import logging
from pathlib import Path
import sys
import warnings

from dotenv import find_dotenv
from loguru import logger


# ==============================================================================
# 0. WARNINGS SÜSPANSİYONU (Konsol Temizliği)
# ==============================================================================
def suppress_annoying_warnings():
    """Çok spesifik, can sıkıcı ve bildiğimiz Python warning'lerini gizler."""
    # Sınıf bazlı filtreleme yerine tamamen string/text bazlı filtreleme daha güvenlidir.
    warnings.filterwarnings("ignore", category=Warning, module="optuna.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="optuna.*")

    # Özellikle OptunaSearchCV için spesifik ExperimentalWarning filtresi
    # Uyarı optuna.integration.sklearn modülünden de fırlatılabilir.
    warnings.filterwarnings("ignore", message=".*OptunaSearchCV is experimental.*")


# ==============================================================================
# 1. RECORD PATCHERS (Kategori Atama ve Manipülasyon)
# ==============================================================================
# Log mesajları (records) handler'lara ulaşmadan *önce* bu sınıftan geçer.
# Bu yapı sayesınde herhangi bir kütüphaneden gelen loglara dinamik olarak
# 'kategori' etiketi (tag) basabilir veya mesajları zenginleştirebiliriz.
class LogPatcher:
    @staticmethod
    def patch_optuna(record):
        """Optuna'dan gelen belirli loglara kategori etiketi ekler."""
        # InterceptHandler'dan geliyorsa 'extra' içinde original_name vardır,
        # ancak bazen loguru üzerinden de direkt basılabilir diye fallback olarak record["name"]'e bakarız.
        logger_name = record["extra"].get("original_name", record["name"])

        # Optuna'nın ana gövdesinden veya trial modülünden gelen Trial mesajlarını yakala
        if logger_name and logger_name.startswith("optuna"):
            if "Trial" in record["message"] and "finished with value:" in record["message"]:
                record["extra"]["category"] = "HPT-Detail"

    @staticmethod
    def patch_mlflow(record):
        """MLflow loglarına kategori etiketi ekler."""
        logger_name = record["extra"].get("original_name", record["name"])

        if logger_name and logger_name.startswith("mlflow"):
            if record["level"].name == "INFO":  # MLflow'un önemli loglarını işaretle
                record["extra"]["category"] = "MLFLOW"

    @classmethod
    def apply(cls, record):
        """Loguru patch mekanizması her log için bu ana fonksiyonu çağırıp manipülasyonları uygular."""
        # Varsayılan olarak her logun bir category anahtarı olsun ki format patlamasın
        if "category" not in record["extra"]:
            record["extra"]["category"] = ""

        cls.patch_optuna(record)
        cls.patch_mlflow(record)


class LogFilter:
    _DISABLED_CATEGORIES = set()

    @classmethod
    def disable_category(cls, category_name: str):
        cls._DISABLED_CATEGORIES.add(category_name)

    @classmethod
    def enable_category(cls, category_name: str):
        cls._DISABLED_CATEGORIES.discard(category_name)

    @classmethod
    def apply(cls, record):
        """İstenmeyen log mesajlarını loguru seviyesinde filtreler (göstermez)."""
        logger_name = record["extra"].get("original_name", record["name"])
        cat = record["extra"].get("category", "")

        # Kullanıcının dinamik olarak kapattığı kategorileri filtrele
        if cat in cls._DISABLED_CATEGORIES:
            return False

        # Optuna'nın 'A new study created in memory...' bilgi mesajını tamamen engelliyoruz
        if (
            logger_name == "optuna.storages._in_memory"
            and record["function"] == "create_new_study"
        ):
            return False

        return True


# API endpoints for external use
disable_log_category = LogFilter.disable_category
enable_log_category = LogFilter.enable_category


# ==============================================================================
# 2. INTERCEPT HANDLER (Standart Logging -> Loguru Köprüsü)
# ==============================================================================
class InterceptHandler(logging.Handler):
    """Standart Python logging mesajlarını yakalayıp Loguru'ya ileten profesyonel hook."""

    def __init__(self, logger_instance):
        super().__init__()
        self._logger = logger_instance

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Doğru stack frame'i bul (logın nereden geldiğini görmek için)
        frame, depth = sys._getframe(2), 2  # default depth

        # Go back in stack until we get out of the `logging` module
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Orijinal logger adını 'extra' dictionary içine koyuyoruz ki patcher'larda kullanabilelim
        self._logger.bind(original_name=record.name).opt(
            depth=depth, exception=record.exc_info
        ).log(level, record.getMessage())


# ==============================================================================
# 3. FORMATTERS (Dinamik Etiket Gösterimi ve Hiyerarşi)
# ==============================================================================
# Kategori bazlı hiyerarşik girinti (indentation) haritası
CATEGORY_INDENT_MAP = {
    "HPT-Detail": "    ",  # HPT detayları bir tab (4 boşluk) içeriden başlar
    "Inner-Run": "",  # Ana seviye
    "DATA": "",  # Ana seviye
    "MLFLOW": "",  # Ana seviye
}


def console_formatter(record):
    """
    Konsol çıktısını dinamik olarak formatlar.
    Eğer loga bir 'category' tag'i eklenmişse bunu renkli bir köşeli parantez içinde gösterir.
    Eklenmemişse standart formatta basar.
    Kategoriye göre dinamik girinti ekler.
    """
    cat = record["extra"].get("category", "")
    indent = CATEGORY_INDENT_MAP.get(cat, "")

    # Zaman ve level artık gösterilmiyor (kullanıcı kapattı)
    # base_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> "
    base_format = indent

    # if cat:
    #     # Kategori varsa köşeli parantez içinde mormsu renkle göster
    #     base_format += "| <m>[{extra[category]}]</m> "

    # Kalan modül ve mesaj kısmını ekle
    # base_format += "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"
    base_format += "<level>{message}</level>\n"
    return base_format


def file_formatter(record):
    """
    Dosya çıktısını dinamik olarak formatlar.
    Eğer loga bir 'category' tag'i eklenmişse bunu gösterir.
    Eklenmemişse boş bırakarak aynı formatı korur.
    Kategoriye göre dinamik girinti ekler.
    """
    cat = record["extra"].get("category", "")
    indent = CATEGORY_INDENT_MAP.get(cat, "")
    cat_str = f" | [{cat}] " if cat else " "

    return (
        indent
        + "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level>"
        + cat_str
        + "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"
    )


# ==============================================================================
# 4. AYARLARI BAŞLAT (Setup)
# ==============================================================================
def setup_logger():
    # 0. Terminal gürültüsünü sustur
    logging.captureWarnings(True)  # Standart warningleri logging'e yönlendir ki biz yakalayalım
    suppress_annoying_warnings()

    # 1. Proje kök dizinini ve log klasörünü bul/oluştur
    try:
        project_root = Path(find_dotenv()).parent
        if str(project_root) == ".":
            project_root = Path(__file__).resolve().parent.parent.parent
    except Exception:
        project_root = Path(__file__).resolve().parent.parent.parent

    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. Loguru'nun varsayılan (sadece konsola basan) ayarını sıfırla
    logger.remove()

    # 3. Patcher'ı logger'a bağla. Artık her log `LogPatcher.apply`'dan geçecek
    patched_logger = logger.patch(LogPatcher.apply)

    # 4. KONSOL ÇIKTISI
    patched_logger.add(sys.stderr, format=console_formatter, level="INFO", filter=LogFilter.apply)

    # 5. GENEL DOSYA ÇIKTISI
    patched_logger.add(
        log_dir / "s2t_fs_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        format=file_formatter,
        level="DEBUG",
        filter=LogFilter.apply,
    )

    # 6. STANDART LOGGING YAKALAMASI (Intercept)
    # Tüm standart Python logging çağrılarını (MLflow, Optuna vb.) Loguru'ya yönlendir
    logging.basicConfig(handlers=[InterceptHandler(patched_logger)], level=0, force=True)

    # Optuna kendi içinde özel bir StreamHandler barındırabiliyor, onu kaldıralım
    # Ayrıca root logger'a (ve dolayıyla loguru'ya) düşebilmesi için propagate'i açalım
    try:
        import optuna

        optuna.logging.disable_default_handler()
        optuna.logging.enable_propagation()
    except ImportError:
        pass

    logging.root.handlers = []
    logging.root.addHandler(InterceptHandler(patched_logger))
    logging.root.setLevel(logging.INFO)

    # Kütüphanelerin minimum native yayınlama seviyeleri
    logging.getLogger("optuna").setLevel(logging.INFO)
    logging.getLogger("optuna.study").setLevel(logging.INFO)
    logging.getLogger("optuna.study.study").setLevel(logging.INFO)

    logging.getLogger("mlflow").setLevel(logging.INFO)
    logging.getLogger("mlflow.tracking").setLevel(logging.INFO)
    logging.getLogger("mlflow.store").setLevel(logging.WARNING)
    logging.getLogger("mlflow.utils").setLevel(logging.WARNING)

    return patched_logger


# Modül import edildiğinde konfigürasyonu çalıştır ve export et
custom_logger = setup_logger()
