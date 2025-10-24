# This script is not well written, since it hardcodes many constants for convenience,
# and it also searches for test log for getting model test accuracy,
# instead of directly evaluating the model.

import argparse
from enum import StrEnum
from pathlib import Path

import torch
from tabulate import tabulate

from neural_pruning.model import LeNet5, LeNet300100, PrunedLeNet5, PrunedLeNet300100


class ModelTypeEnum(StrEnum):
    LeNet5 = "LeNet5"
    LeNet300100 = "LeNet300100"


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Generate comparison table between unpruned and pruned models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[e.value for e in ModelTypeEnum],
        default=ModelTypeEnum.LeNet5,
        help="Type of model to compare (default: LeNet5)",
    )
    return parser.parse_args()


def get_models(model_type: ModelTypeEnum):
    match model_type:
        case ModelTypeEnum.LeNet5:
            unpruned_model = LeNet5()
            pruned_model = PrunedLeNet5()
        case ModelTypeEnum.LeNet300100:
            unpruned_model = LeNet300100()
            pruned_model = PrunedLeNet300100()

    return unpruned_model, pruned_model


def format_table(unpruned_data: dict, pruned_data: dict, metric: str) -> str:
    layers = unpruned_data.keys()
    assert (
        layers == pruned_data.keys()
    ), "Layer mismatch between unpruned and pruned models"

    table_data = []
    headers = [
        "Layer",
        f"{metric} Before Pruning",
        f"{metric} After Pruning",
        "Difference",
        "Reduction %",
    ]
    total_unpruned = 0
    total_pruned = 0

    # Populate table rows
    for layer in layers:
        unpruned_val = unpruned_data[layer]
        pruned_val = pruned_data[layer]
        diff = unpruned_val - pruned_val
        reduction = (diff / unpruned_val * 100) if unpruned_val != 0 else 0

        table_data.append(
            [
                layer,
                f"{unpruned_val:.2f}",
                f"{pruned_val:.2f}",
                f"{diff:.2f}",
                f"{reduction:.1f}%",
            ]
        )

        total_unpruned += unpruned_val
        total_pruned += pruned_val

    # Add summary row
    total_diff = total_unpruned - total_pruned
    total_reduction = (total_diff / total_unpruned * 100) if total_unpruned != 0 else 0
    table_data.append(
        [
            "**TOTAL**",
            f"**{total_unpruned:.2f}**",
            f"**{total_pruned:.2f}**",
            f"**{total_diff:.2f}**",
            f"**{total_reduction:.1f}%**",
        ]
    )

    return tabulate(table_data, headers, tablefmt="github", numalign="right")


def format_summary_table(
    unpruned_model_name: str,
    pruned_model_name: str,
    unpruned_parameters: float,
    pruned_parameters: float,
    unpruned_accuracy: float,
    pruned_accuracy: float,
) -> str:
    headers = ["Model", "Test Accuracy (%)", "Parameters (K)", "Compression Ratio"]
    table_data = [
        [
            unpruned_model_name,
            f"{unpruned_accuracy:.2f}%",
            f"{unpruned_parameters:.2f}",
            "1.00x",
        ],
        [
            pruned_model_name,
            f"{pruned_accuracy:.2f}%",
            f"{pruned_parameters:.2f}",
            f"{unpruned_parameters / pruned_parameters:.2f}x",
        ],
    ]
    return tabulate(table_data, headers, tablefmt="github", numalign="right")


def _retreive_accuracy_from_log(log_path: Path) -> float:
    """Retrieve test accuracy from a log file."""
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "Final Accuracy" in line:
                parts = line.strip().split(":")
                accuracy_str = parts[-1].strip().rstrip("%")
                return float(accuracy_str)

    raise ValueError(f"Key 'Final Accuracy' not found in log file: {log_path}")


def _normalize(data_dict):
    """Divide 1000 to each value in the dictionary."""
    return {k: v / 1000.0 for k, v in data_dict.items()}


def compare_models(
    unpruned_model,
    pruned_model,
    input_size: tuple[int, int, int, int],
    unpruned_log_path: Path,
    pruned_log_path: Path,
) -> str:
    """Compare unpruned and pruned models, return formatted Markdown tables."""
    output = ""
    output += "## Parameter Count Comparison\n\n"
    unpruned_params = _normalize(unpruned_model.per_layer_parameters())
    pruned_params = _normalize(pruned_model.per_layer_parameters())
    output += format_table(unpruned_params, pruned_params, "Parameters (K)")

    output += "\n\n## Model Size Comparison\n\n"
    unpruned_size = _normalize(unpruned_model.per_layer_bytes())
    pruned_size = _normalize(pruned_model.per_layer_bytes())
    output += format_table(unpruned_size, pruned_size, "Size (KB)")

    output += "\n\n## FLOPs Comparison\n\n"
    unpruned_flops = _normalize(unpruned_model.per_layer_flops(input_size))
    pruned_flops = _normalize(pruned_model.per_layer_flops(input_size))
    output += format_table(unpruned_flops, pruned_flops, "FLOPs (K)")

    # Summary Table
    unpruned_total_params = unpruned_model.total_parameters() / 1000.0
    pruned_total_params = pruned_model.total_parameters() / 1000.0
    unpruned_accuracy = _retreive_accuracy_from_log(unpruned_log_path)
    pruned_accuracy = _retreive_accuracy_from_log(pruned_log_path)
    output += "\n\n## Summary\n\n"
    output += format_summary_table(
        unpruned_model.name,
        pruned_model.name,
        unpruned_total_params,
        pruned_total_params,
        unpruned_accuracy,
        pruned_accuracy,
    )

    return output


if __name__ == "__main__":
    MODEL_TYPE = parse_arg().model_type

    # Paths use to load models
    PROJECT_ROOT_DIR = Path(__file__).parent.parent
    CHECKPOINT_DIR = PROJECT_ROOT_DIR / "checkpoints"
    ORIGIN_CHECKPOINT_DIR = CHECKPOINT_DIR / "origin"
    PRUNED_CHECKPOINT_DIR = CHECKPOINT_DIR / "pruned"

    # Path to save output comparison table
    OUTPUT_FILE_PATH = PROJECT_ROOT_DIR / "model_comparison.md"

    # Paths used to retrieve test accuracies
    LOG_DIR = PROJECT_ROOT_DIR / "logs" / "test"

    # Model-specific Paths and input sizes
    ORIGIN_CHECKPOINT_FILENAME = f"{MODEL_TYPE}_epoch10.pt"
    PRUNED_CHECKPOINT_FILENAME = f"Pruned{MODEL_TYPE}_iter3.pt"
    ORIGIN_LOGFILENAME = f"test_{MODEL_TYPE}_epoch10.log"
    PRUNED_LOGFILENAME = f"test_Pruned{MODEL_TYPE}_iter3.log"

    ORIGIN_LOG_PATH = LOG_DIR / ORIGIN_LOGFILENAME
    PRUNED_LOG_PATH = LOG_DIR / PRUNED_LOGFILENAME

    if MODEL_TYPE == ModelTypeEnum.LeNet5:
        INPUT_SIZE = (1, 1, 32, 32)
    else:
        INPUT_SIZE = (1, 1, 28, 28)

    # Load models
    unpruned_model, pruned_model = get_models(MODEL_TYPE)
    unpruned_model.load_state_dict(
        torch.load(ORIGIN_CHECKPOINT_DIR / ORIGIN_CHECKPOINT_FILENAME)
    )
    pruned_model.load_state_dict(
        torch.load(PRUNED_CHECKPOINT_DIR / PRUNED_CHECKPOINT_FILENAME)
    )

    # Compare models and output results
    output = f"# Model Comparison: {MODEL_TYPE}\n\n"
    output += f"**Unpruned Model**: `{ORIGIN_CHECKPOINT_FILENAME}`\n\n"
    output += f"**Pruned Model**: `{PRUNED_CHECKPOINT_FILENAME}`\n\n"
    output += compare_models(
        unpruned_model, pruned_model, INPUT_SIZE, ORIGIN_LOG_PATH, PRUNED_LOG_PATH
    )
    print(output)

    with open(OUTPUT_FILE_PATH, "w") as f:
        f.write(output)

    print(f"\nComparison table saved to {OUTPUT_FILE_PATH}")
