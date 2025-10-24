import json
from dataclasses import asdict, dataclass


def _config_to_str(config):
    config_dict = asdict(config)
    return json.dumps(config_dict, indent=4)


@dataclass
class CommonConfig:
    dataset_dir: str  # Path to dataset directory
    log_dir: str  # Path to logs directory


@dataclass
class TrainConfig(CommonConfig):
    model_type: str  # LeNet5 or LeNet300100
    num_epochs: int  # Number of training epochs
    batch_size: int  # Batch size for training
    learning_rate: float  # Learning rate for optimizer
    weight_decay: float  # Weight decay (L2 regularization) factor
    drop_last: bool  # Whether to drop the last incomplete batch
    model_save_dir: str  # Path to model checkpoints directory

    def __str__(self):
        return _config_to_str(self)


@dataclass
class TestConfig(CommonConfig):
    model_type: str  # LeNet5 or LeNet300100 or PrunedLeNet5 or PrunedLeNet300100
    model_load_path: (
        str  # Path to load pre-trained model checkpoints (full path to file)
    )
    pruned_model_load_path: (
        str  # Path to load pruned model checkpoints (full path to file)
    )

    batch_size: int  # Batch size for training
    drop_last: bool  # Whether to drop the last incomplete batch
    pruned: bool  # Whether the model is pruned

    def __str__(self):
        return _config_to_str(self)


@dataclass
class PruneConfig(CommonConfig):
    model_type: str  # LeNet5 or LeNet300100
    model_load_path: (
        str  # Path to load pre-trained model checkpoints (full path to file)
    )
    num_iterations: int  # Number of pruning-retraining iterations
    num_epochs: int  # Number of retraining epochs after pruning
    batch_size: int  # Batch size for retraining
    learning_rate: float  # Learning rate for retraining
    weight_decay: float  # Weight decay (L2 regularization) factor for retraining
    drop_last: bool  # Whether to drop the last incomplete batch
    model_save_dir: str  # Path to save pruned model checkpoints directory

    def __str__(self):
        return _config_to_str(self)
