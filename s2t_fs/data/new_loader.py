from pathlib import Path

from dotenv import find_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from s2t_fs.utils.logger import custom_logger as logger


def _resolve_path(data_params) -> Path:
    """
    Resolve the parquet path produced by create_parquet.py.

    Priority:
      1. data_params['path']      (explicit override)
      2. data/processed/<dataset>/<subset>_<split>/features.parquet
      3. data/processed/<dataset>.parquet                (legacy fallback)
    """
    project_dir = Path(find_dotenv()).parent

    if data_params.get("path"):
        return project_dir / data_params["path"]

    dataset = data_params["dataset"]
    subset = data_params.get("subset")
    split = data_params.get("split_name", "test")

    if subset:
        return (
            project_dir
            / "data"
            / "processed"
            / dataset
            / f"{subset}_{split}"
            / "features.parquet"
        )

    return project_dir / "data" / "processed" / f"{dataset}.parquet"


def load_and_prepare_data(data_params):
    if data_params.get("source") == "synthetic":
        from s2t_fs.data.synthetic import generate_synthetic_data

        return generate_synthetic_data(data_params)

    path = _resolve_path(data_params)

    logger.bind(category="DATA").debug(
        f"Veri yükleme işlemi başlatıldı. Parametreler: {data_params}. Data path: {path}."
    )

    df = pd.read_parquet(path)

    # --- uid: synthesize if missing ---
    if "uid" not in df.columns:
        df = df.reset_index(drop=True)
        df["uid"] = [f"row_{i}" for i in range(len(df))]

    # --- wer columns: anything ending in _wer ---
    wer_cols = [c for c in df.columns if c.endswith("_wer")]
    df[wer_cols] = df[wer_cols].fillna(1.0)

    # --- feature columns: anything containing _embedding_ (i.e. flattened
    # outputs from create_parquet.py with flattened=true), sorted by alias
    # then by trailing dim index for stable ordering. ---
    def _emb_sort_key(col: str):
        # split into "<alias>_embedding" + "<idx>"
        head, _, tail = col.rpartition("_")
        try:
            return (head, int(tail))
        except ValueError:
            return (col, -1)

    feature_cols = sorted(
        [c for c in df.columns if "_embedding_" in c],
        key=_emb_sort_key,
    )

    if not feature_cols:
        raise ValueError(
            f"No '*_embedding_*' feature columns found in {path}. "
            "Did you run create_parquet with flattened=true?"
        )

    features_df = df[["uid"] + feature_cols].drop_duplicates(subset="uid").set_index("uid")
    wer_df = df[["uid"] + wer_cols].drop_duplicates(subset="uid").set_index("uid")
    common = wer_df.index.intersection(features_df.index)

    X = features_df.loc[common].values.astype(np.float32)
    Y = wer_df.loc[common].values.astype(np.float32)

    seed = data_params["seed"]
    rng = np.random.default_rng(seed)

    if data_params["row_subsample"] < 1.0:
        idx = rng.choice(len(X), size=int(len(X) * data_params["row_subsample"]), replace=False)
        X, Y = X[idx], Y[idx]

    if data_params["feature_subsample"] < 1.0:
        feats = rng.choice(
            X.shape[1], size=int(X.shape[1] * data_params["feature_subsample"]), replace=False
        )
        X = X[:, feats]

    if data_params.get("standard_normalize"):
        X = StandardScaler().fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=data_params["test_size"], random_state=seed
    )

    dataset_stats = {
        "num_features": X.shape[1],
        "num_total_rows": len(X),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "wer_targets": wer_cols,
    }

    logger.bind(category="DATA").debug(
        f"Veri hazırlama işlemi tamamlandı. İstatistikler: {dataset_stats}"
    )

    return X_train, Y_train, X_test, Y_test, dataset_stats
