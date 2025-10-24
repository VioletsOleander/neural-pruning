import logging
from typing import cast

import torch

from neural_pruning.config import TestConfig
from neural_pruning.helper import (
    ModeEnum,
    configuration_prepare_helper,
    data_prepare_helper,
    logger_init_helper,
    model_prepare_helper,
    test_helper,
)
from neural_pruning.model import PrunableModel

if __name__ == "__main__":
    mode = ModeEnum.TEST

    configs = configuration_prepare_helper(mode)
    configs = cast(TestConfig, configs)
    log_path = logger_init_helper(configs, mode)

    logging.info("Starting testing with the following configurations:")
    logging.info(configs)

    dataloader = data_prepare_helper(configs, mode)
    model = model_prepare_helper(configs, mode)
    criterion = torch.nn.CrossEntropyLoss()

    test_helper(cast(PrunableModel, model), dataloader, criterion)
    logging.info(f"Log saved at: {log_path}")
