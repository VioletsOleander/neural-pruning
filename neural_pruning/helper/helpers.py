import argparse
import json
import logging
import tomllib
from pathlib import Path

import torch
from dacite import from_dict

from neural_pruning.config import PruneConfig, TestConfig, TrainConfig
from neural_pruning.model import PrunableModel, PrunedModel

from ._getters import get_dataloader, get_dataset, get_model
from .modes import ModeEnum


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Pruning Helper Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration file. If relative, it is relative to the project root directory.",
    )

    return parser.parse_args()


def configuration_prepare_helper(
    mode: ModeEnum, config_path: str | None = None
) -> TrainConfig | TestConfig | PruneConfig:
    """
    Prepare configuration based on the mode and optional config path.

    Args:
        mode (ModeEnum): Mode of operation.
        config_path (str | None): Path to the configuration file. If None, it will be parsed from command-line arguments.

    Returns:
        TrainConfig | TestConfig | PruneConfig: The prepared configuration dataclass instance.

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    if config_path is None:
        args = _parse_args()
        config_full_path = Path(args.config_path).resolve()
    else:
        config_full_path = Path(config_path).resolve()

    with config_full_path.open("rb") as f:
        configs = tomllib.load(f)

    common_config_dict = configs.get("common", {})
    match mode:
        case ModeEnum.TRAIN:
            config_dict = configs.get("train", {})
            dataclass = TrainConfig
        case ModeEnum.TEST:
            config_dict = configs.get("test", {})
            dataclass = TestConfig
        case ModeEnum.PRUNE:
            config_dict = configs.get("prune", {})
            dataclass = PruneConfig
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

    config_dict.update(common_config_dict)
    config_dataclass = from_dict(dataclass, config_dict)

    return config_dataclass


def _compose_log_file_name(configs, mode: ModeEnum) -> str:
    prefix = mode.name.lower() + "_"
    suffix = ".log"

    match mode:
        case ModeEnum.TRAIN | ModeEnum.PRUNE:
            return prefix + configs.model_type + suffix
        case ModeEnum.TEST:
            stem = (
                Path(configs.model_load_path).stem
                if configs.pruned is False
                else Path(configs.pruned_model_load_path).stem
            )
            return prefix + stem + suffix


def logger_init_helper(
    configs, mode: ModeEnum, log_file_name: str | None = None
) -> str:
    """
    Initialize logging configuration. If log_file_name is not provided, it will be generated based on the mode and model type.

    Args:
        config: Global configuration object. It is expected to have attribute: log_dir.
        mode (ModeEnum): Mode of operation.
        log_file_name (str): Name of the log file, with extension specified (e.g., .log).

    Returns:
        str: Full path to the log file.
    """
    log_dir = Path(configs.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file_name is None:
        log_file_name = _compose_log_file_name(configs, mode)
    log_file = log_dir / log_file_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )

    return str(log_file)


def data_prepare_helper(configs, mode: ModeEnum) -> torch.utils.data.DataLoader:
    """
    Prepare dataset and dataloader based on the configuration.

    Args:
        config: Global configuration object. It is expected to have attributes: dataset_path, model_type, batch_size, drop_last.
        mode (ModeEnum): Mode of operation.

    Returns:
        torch.utils.data.DataLoader: The prepared DataLoader.

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    match mode:
        case ModeEnum.TRAIN | ModeEnum.PRUNE:
            dataset_split = "train"
        case ModeEnum.TEST:
            dataset_split = "test"
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

    dataset = get_dataset(configs.dataset_dir, configs.model_type, dataset_split)
    logging.info(f"Loaded dataset with number of examples: {len(dataset)}.")
    dataloader = get_dataloader(dataset, configs.batch_size, configs.drop_last)
    logging.info(f"Created DataLoader with number of batches: {len(dataloader)}.")

    return dataloader


def model_prepare_helper(configs, mode: ModeEnum) -> torch.nn.Module:
    """
    Initialize model based on the configuration. If mode is TEST or PRUNE, also loads pre-trained model weights. This function will also logs some useful information about the model.

    Args:
        config: Global configuration object. It is expected to have attribute: model_type, model_load_path (for TEST and PRUNE mode).
        mode (ModeEnum): Mode of operation.

    Returns:
        torch.nn.Module: The initialized model.
    """
    model = get_model(configs, mode)
    logging.info(f"Initialized model: {model.name} for {mode} mode.")

    if mode == ModeEnum.TEST or mode == ModeEnum.PRUNE:
        match mode:
            case ModeEnum.PRUNE:
                model_load_path = configs.model_load_path
            case ModeEnum.TEST:
                model_load_path = (
                    configs.model_load_path
                    if configs.pruned is False
                    else configs.pruned_model_load_path
                )
        model.load_state_dict(torch.load(Path(model_load_path)))
        logging.info(f"Loaded model weights from {model_load_path}.")
        logging.info(f"Model sparsity: {model.sparsity}")

    # Report model parameter counts
    per_layer_parameters = model.per_layer_parameters()
    total_parameters = 0
    for value in per_layer_parameters.values():
        total_parameters += value

    logging.info(
        f"Per layer parameters count: {json.dumps(per_layer_parameters, indent=4)}"
    )
    logging.info(f"Total parameters count: {total_parameters}.")

    per_layer_bytes = model.per_layer_bytes()
    total_bytes = 0
    for value in per_layer_bytes.values():
        total_bytes += value

    # Report model size in kilobytes
    per_layer_kilobytes = {key: value / 1024 for key, value in per_layer_bytes.items()}
    total_kilobytes = total_bytes / 1024
    logging.info(
        f"Per layer parameter sizes (KB): {json.dumps(per_layer_kilobytes, indent=4)}"
    )
    logging.info(f"Total model size (KB): {total_kilobytes:.2f}.")

    return model


def optim_prepare_helper(model: torch.nn.Module, configs) -> torch.optim.Optimizer:
    """
    Initialize SGD optimizer based on the model and configuration.

    Args:
        model (torch.nn.Module): The model for which the optimizer is to be created.
        configs: Global configuration object. It is expected to have attributes: learning_rate, weight_decay

    Returns:
        torch.optim.Optimizer: The initialized SGD optimizer.
    """
    # weight decay is effectively L2 regularization
    optim = torch.optim.SGD(
        model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay
    )
    logging.info(
        f"Initialized SGD optimizer with learning rate: {configs.learning_rate}."
    )
    return optim


def _save_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)


def train_helper(
    model: PrunableModel,
    dataloader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    configs,
    save_model: bool = True,
) -> None:
    """
    Train a prunable model using the provided dataloader, optimizer, and criterion. Logs training progress and saves checkpoints.

    Args:
        model (PrunableModel): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        optim (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        configs: Global configuration object. It is expected to have attributes: num_epochs, model_save_dir.

    Returns:
        None
    """
    logging.info("Starting training...")
    for epoch in range(configs.num_epochs):
        model.train()
        logging.info(f"Starting epoch {epoch + 1}/{configs.num_epochs}")
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["image"]
            targets = batch["label"]

            optim.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            model.mask_gradients()
            optim.step()

            if (batch_idx + 1) % 100 == 0:
                logging.info(
                    f"Epoch [{epoch + 1}/{configs.num_epochs}], "
                    f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        logging.info(f"Epoch {epoch + 1} completed.")
        if save_model:
            ckpt_path = f"{configs.model_save_dir}/{model.name}_epoch{epoch + 1}.pt"
            _save_checkpoint(model, ckpt_path)
            logging.info(f"Saved model checkpoint to {ckpt_path}.")

    logging.info("Training completed.")


def prune_helper(
    model: PrunedModel,
    dataloader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    configs,
) -> None:
    logging.info("Starting pruning and retraining process...")
    for iteration in range(configs.num_iterations):
        logging.info(
            f"Pruning-Retraining Iteration {iteration + 1}/{configs.num_iterations}"
        )

        thresholds = model.prune_thresholds
        logging.info("Determined layer-wise pruning thresholds:")
        for name, threshold in thresholds.items():
            logging.info(f"  {name}: {threshold:.6f}")

        # Pruning
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in thresholds and thresholds[name] > 0.0:
                    mask = param.abs() >= thresholds[name]
                    model.masks[name] = model.masks[name] & mask
            model.prune_parameters()
            logging.info(f"Pruned model, current sparsity: {model.sparsity:.2%}")

        # Retraining
        logging.info(f"Starting retraining for {configs.num_epochs} epochs...")
        train_helper(model, dataloader, optim, criterion, configs, save_model=False)

        _save_checkpoint(
            model, f"{configs.model_save_dir}/{model.name}_iter{iteration + 1}.pt"
        )

    logging.info("Pruning and retraining process completed.")


def test_helper(
    model: PrunableModel,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
) -> None:
    """
    Test a prunable model using the provided dataloader and criterion. Logs testing progress and accuracy.

    Args:
        model (PrunableModel): The model to be tested.
        dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
        criterion (torch.nn.Module): Loss function.

    Returns:
        None
    """
    total = 0
    correct = 0
    total_loss = 0.0

    logging.info("Starting testing...")
    with torch.no_grad():
        model.eval()
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["image"]
            targets = batch["label"]

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predicted = torch.argmax(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                logging.info(
                    f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {100 * correct / total:.2f}%"
                )

    average_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    logging.info(
        f"Testing completed. Average Loss: {average_loss:.4f}, Final Accuracy: {accuracy:.2f}%"
    )
