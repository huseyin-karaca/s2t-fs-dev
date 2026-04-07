import hydra
from omegaconf import DictConfig

from s2t_fs.data.config import from_hydra


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    ds_cfg = from_hydra(cfg)
    print(f"⬇️  Fetching {ds_cfg.name} / {ds_cfg.subset} / {ds_cfg.split_name}")
    ds_cfg.load(cache_dir=cfg.data.paths.raw_dir)
    print("✅ cached")


if __name__ == "__main__":
    main()
