from pathlib import Path

from dotenv import find_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from s2t_fs.utils.logger import custom_logger as logger


def load_and_prepare_data(data_params):
    if data_params.get("source") == "synthetic":
        from s2t_fs.data.synthetic import generate_synthetic_data

        return generate_synthetic_data(data_params)

    project_dir = Path(find_dotenv()).parent

    # 2. Veri yolunu kök dizinden itibaren dinamik olarak inşa et
    # pathlib, işletim sistemi ne olursa olsun doğru ayırıcıları kendi belirler
    path = project_dir / "data" / "processed" / f"{data_params['dataset']}.parquet"

    logger.bind(category="DATA").debug(
        f"Veri yükleme işlemi başlatıldı. Parametreler: {data_params}. Data path: {path}."
    )

    df = pd.read_parquet(path)

    wer_cols = [c for c in df.columns if c.startswith("wer_")]
    df[wer_cols] = df[wer_cols].fillna(1.0)

    feature_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
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
    }

    logger.bind(category="DATA").debug(
        f"Veri hazırlama işlemi tamamlandı. İstatistikler: {dataset_stats}"
    )

    return X_train, Y_train, X_test, Y_test, dataset_stats
