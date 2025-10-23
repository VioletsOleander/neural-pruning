import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD

from deep_compression.utils import get_dataloader, get_dataset, get_model, parse_configs


def configure_logger(log_path: str, log_file_name: str = "train.log") -> None:
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


def save_checkpoint(model: nn.Module, epoch: int, checkpoint_dir: str) -> None:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{model.__class__.__name__}_epoch{epoch + 1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"Saved model checkpoint to {checkpoint_path}.")


if __name__ == "__main__":
    # Parse configurations
    configs = parse_configs()

    common_config = configs["common"]
    log_path = common_config["log_path"]
    dataset_path = common_config["dataset_path"]

    train_config = configs["train"]
    learning_rate = train_config["learning_rate"]
    weight_decay = train_config["weight_decay"]
    num_epochs = train_config["num_epochs"]
    model_type = train_config["model_type"]
    batch_size = train_config["batch_size"]
    drop_last = train_config["drop_last"]
    checkpoint_dir = train_config["model_save_path"]

    # Prepare
    configure_logger(log_path)

    logging.info("Starting training with the following configurations:")
    logging.info(json.dumps(configs, indent=4))

    dataset = get_dataset(dataset_path, model_type, mode="train")
    logging.info(f"Loaded dataset with number of examples: {len(dataset)}.")
    dataloader = get_dataloader(dataset, batch_size, drop_last)
    logging.info(f"Created DataLoader with number of batches: {len(dataloader)}.")

    model = get_model(model_type)
    logging.info(
        f"Initialized model: {model.__class__.__name__} with {model.parameter_count()} parameters."
    )
    logging.info("Parameter dtypes and sizes for each layer: ")
    for name, param in model.named_parameters():
        logging.info(f"  {name}: {param.dtype}: {param.size()}")
    logging.info(
        f"Total model size in bytes: {model.total_bytes()} bytes = {model.total_bytes() / 1024:.2f} KB."
    )

    # weight decay is effectively L2 regularization
    optim = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logging.info(f"Initialized SGD optimizer with learning rate: {learning_rate}.")
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    logging.info("Starting training...")
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["image"]
            targets = batch["label"]

            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()

            if (batch_idx + 1) % 10 == 0:
                logging.info(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        save_checkpoint(model, epoch, checkpoint_dir)
        logging.info(f"Epoch {epoch + 1} completed and checkpoint saved.")

    logging.info("Training completed.")
