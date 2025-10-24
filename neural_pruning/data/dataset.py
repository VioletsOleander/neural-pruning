from typing import Literal, cast

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset

MEAN = 0.1307
STD = 0.3081


def _transform_dtype(
    examples: dict[str, list[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    examples["image"] = [img.to(torch.float32) for img in examples["image"]]
    examples["label"] = [label.to(torch.int64) for label in examples["label"]]

    return examples


def _normalize_images(
    examples: dict[str, list[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    examples["image"] = [(img / 255.0 - MEAN) / STD for img in examples["image"]]
    return examples


def _padding_images(
    examples: dict[str, list[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    padded_images = []
    for img in examples["image"]:
        # Pad 28x28 images to 32x32 for LeNet5
        padded_img = torch.nn.functional.pad(img, (2, 2, 2, 2), "constant", 0)
        padded_images.append(padded_img)
    examples["image"] = padded_images
    return examples


class MNISTDataset(Dataset):
    hf_dataset: HFDataset

    def __init__(
        self, dataset_dir: str, model_type: str, split: Literal["train", "test"]
    ):
        self.hf_dataset = cast(
            HFDataset,
            load_dataset(dataset_dir, split=split),
        )

        self.hf_dataset = self.hf_dataset.with_format("torch")
        self.hf_dataset = self.hf_dataset.map(
            _transform_dtype, batched=True, batch_size=256
        )

        if split == "train":
            self.hf_dataset = self.hf_dataset.map(
                _normalize_images, batched=True, batch_size=256
            )

        if model_type == "LeNet5":
            self.hf_dataset = self.hf_dataset.map(
                _padding_images, batched=True, batch_size=256
            )

    def __len__(self):
        return len(self.hf_dataset)

    # the return item is assumed to be a dict with 'image' and 'label' keys
    def __getitem__(self, idx):
        return self.hf_dataset[idx]
