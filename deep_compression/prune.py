import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD

from deep_compression.utils import (
    configure_logger,
    get_dataloader,
    get_dataset,
    get_pruned_model,
    parse_configs,
    save_checkpoint,
)

if __name__ == "__main__":
    # Parse configurations
    configs = parse_configs()

    common_config = configs["common"]
    log_path = common_config["log_path"]
    dataset_path = common_config["dataset_path"]

    prune_config = configs["prune"]

    model_type = prune_config["model_type"]
    batch_size = prune_config["batch_size"]
    drop_last = prune_config["drop_last"]
    model_load_path = prune_config["model_load_path"]
    iteration = prune_config["iteration"]
    num_epochs = prune_config["num_epochs"]
    learning_rate = prune_config["learning_rate"]
    weight_decay = prune_config["weight_decay"]
    model_save_path = prune_config["model_save_path"]

    # Prepare logging
    ckpt_name = Path(model_load_path).stem
    log_file_path = configure_logger(log_path, log_file_name=f"prune_{ckpt_name}.log")

    logging.info("Starting pruning with the following configurations:")
    logging.info(json.dumps(configs, indent=4))

    # Prepare for pruning
    model = get_pruned_model(model_type)
    model.load_state_dict(torch.load(Path(model_load_path)))
    logging.info(
        f"Loaded pre-trained model: {model.name} with {model.parameter_count()} parameters",
    )
    logging.info("Parameter dtypes and sizes for each layer: ")
    for name, param in model.named_parameters():
        logging.info(f"  {name}: {param.dtype}: {param.size()}")
    logging.info(
        f"Total model size in bytes: {model.total_bytes()} bytes = {model.total_bytes() / 1024:.2f} KB."
    )

    # Prepare for retraining
    dataset = get_dataset(dataset_path, model_type, mode="train")
    logging.info(f"Loaded dataset with number of examples: {len(dataset)}.")
    dataloader = get_dataloader(dataset, batch_size, drop_last)
    logging.info(f"Created DataLoader with number of batches: {len(dataloader)}.")

    optim = SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    logging.info(f"Initialized SGD optimizer with learning rate: {learning_rate}.")
    criterion = nn.CrossEntropyLoss()

    # Pruning and retraining loop
    model.train()
    logging.info("Starting pruning and retraining process...")
    for it in range(iteration):
        logging.info(f"Pruning-Retraining Iteration {it + 1}/{iteration}")

        # Determine pruning thresholds
        thresholds = model.get_prune_thresholds()
        logging.info("Determined layer-wise pruning thresholds:")
        for name, threshold in thresholds.items():
            logging.info(f"  {name}: {threshold:.6f}")

        # Apply pruning
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in thresholds and thresholds[name] > 0.0:
                    mask = param.abs() >= thresholds[name]
                    model.masks[name] = model.masks[name] & mask
            model.apply_pruning()
            logging.info(
                f"Applied pruning to the model, current sparsity: {model.sparsity:.2%}"
            )

        # Retrain the pruned model
        logging.info(f"Starting retraining for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                images, labels = batch["image"], batch["label"]

                optim.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                model.apply_gradient_masking()

                optim.step()

                running_loss += loss.item()

                if (batch_idx + 1) % 100 == 0:
                    logging.info(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = running_loss / len(dataloader)
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

        # Save checkpoint after each iteration
        save_checkpoint(model, f"{model_save_path}/{model.name}_it{it + 1}.pt")

    logging.info("Pruning and retraining process completed.")
    logging.info(f"Log saved to {log_file_path}")
