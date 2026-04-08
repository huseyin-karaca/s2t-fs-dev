import os

import hydra
import numpy as np
import librosa
from datasets import Audio
from omegaconf import DictConfig

from s2t_fs.data.config import from_hydra as data_from_hydra


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    ds_cfg = data_from_hydra(cfg)
    n_mfcc = cfg.get("n_mfcc", 13)

    out_dir = os.path.join(cfg.data.paths.interim_dir, f"mfcc{n_mfcc}")
    if os.path.exists(out_dir) and not cfg.overwrite:
        print(f"⏭  exists, skipping: {out_dir}")
        return

    print(f"🚀 {ds_cfg.name}/{ds_cfg.subset} ← mfcc{n_mfcc}")
    ds = ds_cfg.load(cache_dir=cfg.data.paths.raw_dir)
    ds = ds.cast_column("audio", Audio(sampling_rate=ds_cfg.sampling_rate))

    emb_col = f"mfcc{n_mfcc}_embedding"

    def proc(batch):
        out = {emb_col: []}
        for i in range(len(batch["audio"])):
            a = batch["audio"][i]
            mfcc = librosa.feature.mfcc(
                y=np.asarray(a["array"], dtype=np.float32),
                sr=a["sampling_rate"],
                n_mfcc=n_mfcc,
            )
            # mean over time -> fixed-size embedding
            out[emb_col].append(mfcc.mean(axis=1).tolist())
        return out

    feats = ds.map(
        proc,
        batched=True,
        batch_size=cfg.batch_size,
        remove_columns=ds.column_names,
        desc=f"mfcc{n_mfcc} extraction",
    )
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    feats.save_to_disk(out_dir)
    print(f"✅ {out_dir}")


if __name__ == "__main__":
    main()
