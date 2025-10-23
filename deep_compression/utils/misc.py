import argparse
import logging
import tomllib
from pathlib import Path

import torch
from torch import nn


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Compression Utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")

    return parser.parse_args()


def parse_configs() -> dict:
    args = _parse_args()
    config_path = args.config_path

    with open(config_path, "rb") as f:
        configs = tomllib.load(f)

    return configs


def configure_logger(log_path: str, log_file_name: str) -> str:
    Path(log_path).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path) / log_file_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )

    return str(log_file)


def save_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    logging.info(f"Saved model checkpoint to {ckpt_path}.")
