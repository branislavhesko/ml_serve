import dataclasses

from omegaconf import DictConfig, OmegaConf


@dataclasses.dataclass
class ModelConfig:
    model_name: str

    def to_yaml(self, path: str) -> str:
        return OmegaConf.save(dataclasses.asdict(self), path)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        return cls(**OmegaConf.load(path))
    

if __name__ == "__main__":
    config = ModelConfig(model_name="example_model")
    print(config.to_yaml("example_config.yaml"))
    print(ModelConfig.from_yaml("example_config.yaml"))
