import logging
from typing import cast

from torch import nn

from neural_pruning.config import TrainConfig
from neural_pruning.helper import (
    ModeEnum,
    configuration_prepare_helper,
    data_prepare_helper,
    logger_init_helper,
    model_prepare_helper,
    optim_prepare_helper,
    train_helper,
)
from neural_pruning.model import PrunableModel

if __name__ == "__main__":
    mode = ModeEnum.TRAIN

    configs = configuration_prepare_helper(mode)
    configs = cast(TrainConfig, configs)
    log_path = logger_init_helper(configs, mode)

    logging.info("Starting training with the following configurations:")
    logging.info(configs)

    dataloader = data_prepare_helper(configs, mode)
    model = model_prepare_helper(configs, mode)
    optim = optim_prepare_helper(model, configs)
    criterion = nn.CrossEntropyLoss()

    train_helper(cast(PrunableModel, model), dataloader, optim, criterion, configs)
    logging.info(f"Log saved at: {log_path}")
