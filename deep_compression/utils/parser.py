import argparse
import tomllib


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
