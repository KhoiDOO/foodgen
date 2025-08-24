import os
import argparse

import models
import trainer

from omegaconf import OmegaConf

from utils.config import ExperimentConfig, load_config, dump_config

from accelerate.utils import DistributedDataParallelKwargs

def main(args, extras):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    print(f"Running with {n_gpus} GPU(s): {', '.join(selected_gpus)}")

    main_model = getattr(models, cfg.model)(**cfg.model_config)

    # Total number of parameters
    total_params = sum(p.numel() for p in main_model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Result dir
    print(f"Result dir: {cfg.trial_dir}")

    main_trainer = getattr(trainer, cfg.trainer)(
        **{"vae": main_model, "results_folder": cfg.trial_dir}, **cfg.trainer_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    # group = parser.add_mutually_exclusive_group(required=True)

    args, extras = parser.parse_known_args()

    main(args, extras)