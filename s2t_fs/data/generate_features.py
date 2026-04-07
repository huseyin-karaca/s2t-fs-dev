import os
import re
import importlib

import hydra
import torch
import evaluate
from datasets import Audio
from omegaconf import DictConfig

from s2t_fs.data.config import from_hydra as data_from_hydra


def _resolve_model(cfg: DictConfig):
    mod = importlib.import_module(cfg.base_model.module)
    return getattr(mod, cfg.base_model.var)


def _clean(t) -> str:
    return re.sub(r"[^\w\s]", "", str(t or "").lower()).strip()


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    ds_cfg = data_from_hydra(cfg)
    model_cfg = _resolve_model(cfg)
    text_col = cfg.data.source.text_column

    out_dir = os.path.join(cfg.data.paths.interim_dir, model_cfg.alias)
    if os.path.exists(out_dir) and not cfg.overwrite:
        print(f"⏭  exists, skipping: {out_dir}")
        return

    print(f"🚀 {ds_cfg.name}/{ds_cfg.subset} ← {model_cfg.alias}")
    ds = ds_cfg.load(cache_dir=cfg.data.paths.raw_dir)
    ds = ds.cast_column("audio", Audio(sampling_rate=ds_cfg.sampling_rate))

    model, processor = model_cfg.load(device=device)
    wer = evaluate.load("wer")

    emb_col = f"{model_cfg.alias}_embedding"
    txt_col = f"{model_cfg.alias}_transcription"
    wer_col = f"{model_cfg.alias}_wer"

    def proc(batch):
        out = {emb_col: [], txt_col: [], wer_col: []}
        for i in range(len(batch["audio"])):
            a = batch["audio"][i]
            tgt = _clean(batch[text_col][i])
            res = model_cfg.predict(
                model, processor, a["array"], a["sampling_rate"], device
            )
            pred = _clean(res["transcription"])
            out[emb_col].append(res["embedding"])
            out[txt_col].append(pred)
            out[wer_col].append(
                wer.compute(predictions=[pred], references=[tgt]) if tgt else 0.0
            )
        return out

    feats = ds.map(
        proc,
        batched=True,
        batch_size=cfg.batch_size,
        remove_columns=ds.column_names,
        desc=f"{model_cfg.alias} inference",
    )
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    feats.save_to_disk(out_dir)
    print(f"✅ {out_dir}")


if __name__ == "__main__":
    main()
