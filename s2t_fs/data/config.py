from dataclasses import dataclass
from datasets import load_dataset
from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    name: str
    subset: str
    split_name: str
    revision: str = "refs/convert/parquet"
    text_column: str = "text"
    sampling_rate: int = 16000

    @property
    def data_files(self):
        return {self.split_name: f"{self.subset}/{self.split_name}/*.parquet"}

    def load(self, cache_dir: str = "data/raw"):
        return load_dataset(
            self.name,
            revision=self.revision,
            data_files=self.data_files,
            cache_dir=cache_dir,
            split=self.split_name,
        )


def from_hydra(cfg: DictConfig) -> "DatasetConfig":
    s = cfg.data.source
    return DatasetConfig(
        name=s.name,
        subset=s.subset,
        split_name=s.split_name,
        revision=s.revision,
        text_column=s.text_column,
        sampling_rate=s.sampling_rate,
    )
