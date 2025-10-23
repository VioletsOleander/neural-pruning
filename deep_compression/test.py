import json
import logging
from pathlib import Path

import torch

from deep_compression.utils import (
    configure_logger,
    get_dataloader,
    get_dataset,
    get_model,
    get_pruned_model,
    parse_configs,
)

if __name__ == "__main__":
    # Parse configurations
    configs = parse_configs()

    common_config = configs["common"]
    log_path = common_config["log_path"]
    dataset_path = common_config["dataset_path"]

    test_config = configs["test"]
    model_type = test_config["model_type"]
    batch_size = test_config["batch_size"]
    drop_last = test_config["drop_last"]
    model_load_path = test_config["model_load_path"]
    pruned = test_config["pruned"]

    # Prepare
    ckpt_name = model_load_path.split("/")[-1].replace(".pt", "")
    log_file_path = configure_logger(log_path, log_file_name=f"test_{ckpt_name}.log")

    logging.info("Starting testing with the following configurations:")
    logging.info(json.dumps(configs, indent=4))

    dataset = get_dataset(dataset_path, model_type, mode="test")
    logging.info(f"Loaded dataset with number of examples: {len(dataset)}.")
    dataloader = get_dataloader(dataset, batch_size, drop_last)
    logging.info(f"Created DataLoader with number of batches: {len(dataloader)}.")

    logging.info(f"Loading pre-trained model from {model_load_path}")

    model = get_model(model_type) if not pruned else get_pruned_model(model_type)
    model.load_state_dict(torch.load(Path(model_load_path)))

    logging.info(
        f"Initialized model: {model.name} with {model.parameter_count()} effective parameters."
    )
    if pruned:
        logging.info(f"Model sparsity: {model.sparsity}")

    logging.info("Parameter dtypes and sizes for each layer: ")
    for name, param in model.named_parameters():
        logging.info(f"  {name}: {param.dtype}: {param.size()}")
    logging.info(
        f"Total effective model size in bytes: {model.total_bytes()} bytes = {model.total_bytes() / 1024:.2f} KB."
    )

    # Test
    model.eval()
    logging.info("Starting testing...")

    accurate = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, labels = batch["image"], batch["label"]
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            accurate += (predicted == labels).sum().item()

            accuracy = float(accurate) / total * 100

            if (batch_idx + 1) % 10 == 0:
                logging.info(
                    f"Processed {batch_idx + 1}/{len(dataloader)} batches. "
                    f"Current Accuracy: {accuracy:.2f}%"
                )

    logging.info(f"Final Accuracy: {accuracy:.2f}%")
    logging.info(f"Log saved to {log_file_path}")
