import os
import argparse

import models
import trainer
import torch

from omegaconf import OmegaConf

from utils.config import ExperimentConfig, load_config, dump_config

from accelerate.utils import DistributedDataParallelKwargs

def main(args, extras):

    print(extras)

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)

    main_model = getattr(models, cfg.model)(**cfg.model_config)

    # Total number of parameters
    total_params = sum(p.numel() for p in main_model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Result dir
    print(f"Result dir: {cfg.trial_dir}")

    main_trainer = getattr(trainer, cfg.trainer)(
        **{"vae": main_model, "results_folder": cfg.trial_dir}, **cfg.trainer_config)

    if args.train:
        main_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")

    # group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run testing")

    args, extras = parser.parse_known_args()

    main(args, extras)