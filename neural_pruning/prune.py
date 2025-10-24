"""
This script performs model pruning and retraining to an already pre-trained model, using specified configurations.
Therefore, it expects to load a pre-trained model, applies pruning based on layer-wise thresholds, and retrains the pruned model.
"""

import logging
from typing import cast

from torch import nn

from neural_pruning.config import PruneConfig
from neural_pruning.helper import (
    ModeEnum,
    configuration_prepare_helper,
    data_prepare_helper,
    logger_init_helper,
    model_prepare_helper,
    optim_prepare_helper,
    prune_helper,
)
from neural_pruning.model import PrunedModel

if __name__ == "__main__":
    mode = ModeEnum.PRUNE

    configs = configuration_prepare_helper(mode)
    configs = cast(PruneConfig, configs)
    log_path = logger_init_helper(configs, mode)

    logging.info("Starting pruning with the following configurations:")
    logging.info(configs)

    dataloader = data_prepare_helper(configs, mode)
    model = model_prepare_helper(configs, mode)
    optim = optim_prepare_helper(model, configs)
    criterion = nn.CrossEntropyLoss()

    prune_helper(cast(PrunedModel, model), dataloader, optim, criterion, configs)
    logging.info(f"Log saved at: {log_path}")
