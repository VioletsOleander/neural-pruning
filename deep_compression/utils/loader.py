from torch.utils.data import DataLoader

from deep_compression.data import MNISTDataset
from deep_compression.model import LeNet5, LeNet300100


def get_dataset(dataset_path: str, model_type: str, mode: str) -> MNISTDataset:
    dataset = MNISTDataset(dataset_path, model_type, mode)

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


def get_model(model_type: str) -> LeNet5 | LeNet300100:
    if model_type == "LeNet5":
        model = LeNet5()
    elif model_type == "LeNet300100":
        model = LeNet300100()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model
