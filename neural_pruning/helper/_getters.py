from typing import Literal

from torch.utils.data import DataLoader

from neural_pruning.data import MNISTDataset
from neural_pruning.model import LeNet5, LeNet300100, PrunedLeNet5, PrunedLeNet300100

from .modes import ModeEnum


def get_dataset(
    dataset_dir: str, model_type: str, split: Literal["train", "test"]
) -> MNISTDataset:
    dataset = MNISTDataset(dataset_dir, model_type, split)

    return dataset


def get_dataloader(
    dataset: MNISTDataset, batch_size: int, drop_last: bool
) -> DataLoader:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    return dataloader


def _get_unpruned_model(model_type: str) -> LeNet5 | LeNet300100:
    match model_type:
        case "LeNet5":
            model = LeNet5()
        case "LeNet300100":
            model = LeNet300100()
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")
    return model


def _get_pruned_model(model_type: str) -> PrunedLeNet5 | PrunedLeNet300100:
    match model_type:
        case "LeNet5":
            model = PrunedLeNet5()
        case "LeNet300100":
            model = PrunedLeNet300100()
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")
    return model


def get_model(
    configs,
    mode: ModeEnum,
) -> LeNet5 | LeNet300100 | PrunedLeNet5 | PrunedLeNet300100:
    match mode:
        case ModeEnum.TRAIN:
            model = _get_unpruned_model(configs.model_type)
        case ModeEnum.PRUNE:
            model = _get_pruned_model(configs.model_type)
        case ModeEnum.TEST:
            if configs.pruned:
                model = _get_pruned_model(configs.model_type)
            else:
                model = _get_unpruned_model(configs.model_type)
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

    return model
