import os
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob

from omegaconf import OmegaConf, DictConfig
from typing import Optional, Any, Union

from sympy import false

@dataclass
class ExperimentConfig:
    name: str = "default"
    exp_root_dir: str = "outputs"

    ### these shouldn't be set manually
    exp_dir: str = ""
    trial_dir: str = ""
    n_gpus: int = 1
    seed: int = 42
    ###

    model: str = ""
    model_config: dict = field(default_factory=dict)

    trainer: str = ""
    trainer_config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.exp_root_dir is None:
            self.exp_root_dir = "outputs"
        os.makedirs(self.exp_root_dir, exist_ok=True)

        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        os.makedirs(self.exp_dir, exist_ok=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        trial_dir = os.path.join(self.exp_dir, now)

        self.trial_dir = os.path.join(self.exp_dir, now)
        os.makedirs(self.trial_dir, exist_ok=True)

def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> ExperimentConfig:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg