import os
import importlib

import hydra
import pandas as pd
from datasets import load_from_disk
from omegaconf import DictConfig, ListConfig

from s2t_fs.data.config import from_hydra as data_from_hydra


def _resolve_alias(module: str, var: str) -> str:
    return getattr(importlib.import_module(module), var).alias


@hydra.main(version_base=None, config_path="../../configs", config_name="create_parquet")
def main(cfg: DictConfig) -> None:
    ds_cfg = data_from_hydra(cfg)
    interim_dir = cfg.data.paths.interim_dir

    # --- collect feature dirs to merge ---
    aliases: list[str] = []

    # base models: list of base_model config nodes (each has module + var)
    for bm in cfg.base_models:
        aliases.append(_resolve_alias(bm.module, bm.var))

    # mfcc variants: list of n_mfcc ints
    mfcc_aliases = [f"mfcc{n}" for n in cfg.mfcc]

    merged: pd.DataFrame | None = None

    for alias in aliases + mfcc_aliases:
        sub = os.path.join(interim_dir, alias)
        if not os.path.exists(sub):
            print(f"⚠️  missing, skipping: {sub}")
            continue
        print(f"📥 loading {sub}")
        df = load_from_disk(sub).to_pandas()

        if cfg.flattened:
            # drop transcription columns
            df = df.drop(
                columns=[c for c in df.columns if c.endswith("_transcription")],
                errors="ignore",
            )
            # expand list-valued embedding columns into <col>_0, <col>_1, ...
            for col in [c for c in df.columns if c.endswith("_embedding")]:
                expanded = pd.DataFrame(
                    df[col].tolist(),
                    index=df.index,
                ).add_prefix(f"{col}_")
                df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

        merged = df if merged is None else pd.concat([merged, df], axis=1)

    if merged is None:
        print("❌ no interim features found — nothing to write")
        return

    # --- write out ---
    out_dir = os.path.join(
        cfg.processed_dir,
        ds_cfg.name.replace("/", "_"),
        f"{ds_cfg.subset}_{ds_cfg.split_name}",
    )
    os.makedirs(out_dir, exist_ok=True)

    pq_path = os.path.join(out_dir, "features.parquet")
    merged.to_parquet(pq_path, index=False)
    print(f"✅ {pq_path}  ({len(merged)} rows × {len(merged.columns)} cols)")

    if cfg.write_csv:
        csv_path = os.path.join(out_dir, "features.csv")
        merged.to_csv(csv_path, index=False)
        print(f"✅ {csv_path}")


if __name__ == "__main__":
    main()
