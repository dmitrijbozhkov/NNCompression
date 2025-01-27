from collections import namedtuple
from dataclasses import dataclass
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torchvision import transforms, datasets as d
from typing import Tuple
from pathlib import Path
import numpy as np

data_loaders = namedtuple("dataset", ["train_loader", "test_loader", "valid_loader"])

def _dataset_infer_input(dataset):
    """
    Infer dataset input based on first item
    """
    return dataset[0][0].shape

@dataclass
class Dataset:
    """
    Dataset with metadata
    """

    data_loaders: data_loaders

    # Metadata
    task_type: str
    train_size: int
    valid_size: int
    test_size: int
    batch_size: int
    input_shape: Tuple[int, ...]
    output_num: int
    is_batch_first: bool

    @classmethod
    def get_dataset(cls, runner_config: dict):
        """
        Initialize dataset from runner config

        :param runner_config: Runner configuration object
        :returns: Initialized Dataset object
        """
        transform = transforms.ToTensor()
        train_dataset = None
        valid_dataset = None
        test_dataset = None
        data_folder = Path(__file__).parent.parent / "data"
        train_size = 0
        valid_size = 0
        test_size = 0

        if runner_config["dataset"] == 'cifar10':
            train_dataset = d.CIFAR10(
                root=str(data_folder),
                train=True,
                download=True,
                transform=transform)
            test_dataset = d.CIFAR10(
                root=str(data_folder),
                train=False,
                download=True,
                transform=transform
            )
            task_type = "classification"
            train_size = len(train_dataset)
            test_size = len(test_dataset)
            input_shape = _dataset_infer_input(train_dataset)
            output_num = 10

        if runner_config["dataset"] == 'MNIST':
            train_dataset = d.MNIST(
                root=str(data_folder),
                train=True,
                download=True,
                transform=transform
            )
            test_dataset = d.MNIST(
                root=str(data_folder),
                train=False,
                download=True,
                transform=transform
            )
            task_type = "classification"
            train_size = len(train_dataset)
            test_size = len(test_dataset)
            input_shape = _dataset_infer_input(train_dataset)
            output_num = 10

        if runner_config["dataset"] == 'FashionMNIST':
            train_dataset = d.FashionMNIST(
                root=str(data_folder),
                train=True,
                download=True,
                transform=transform
            )
            test_dataset = d.FashionMNIST(
                root=str(data_folder),
                train=False,
                download=True,
                transform=transform
            )
            task_type = "classification"
            train_size = len(train_dataset)
            test_size = len(test_dataset)
            input_shape = _dataset_infer_input(train_dataset)
            output_num = 10

        if runner_config["dataset"] == "cifar100":
            train_dataset = d.CIFAR100(
                root=str(data_folder),
                train=True,
                transform=transform)
            test_dataset = d.CIFAR100(
                root=str(data_folder),
                train=False,
                transform=transform
            )
            task_type = "classification"
            train_size = len(train_dataset)
            test_size = len(test_dataset)
            input_shape = _dataset_infer_input(train_dataset)
            output_num = 100
        if not runner_config["is_dataset_canon_split"]:
            assert (
                runner_config["dataset_train_split"] +
                runner_config["dataset_valid_split"] +
                runner_config["dataset_test_split"]
            ) == 1, "dataset_split ratios should sum up to 1"
            assert (
                runner_config["dataset_train_split"] > 0 and
                runner_config["dataset_train_split"] < 1
            ), "dataset_train_split should be within 0 and 1"
            assert (
                runner_config["dataset_test_split"] > 0 and
                runner_config["dataset_test_split"] < 1
            ), "dataset_test_split should be within 0 and 1"

            datasets = [
                train_dataset,
                valid_dataset,
                test_dataset
            ]
            full_dataset = ConcatDataset([d for d in datasets if d is not None])

            total_num = len(full_dataset)
            indices = list(range(total_num))

            train_split = int(np.floor(total_num * runner_config["dataset_train_split"]))
            if runner_config["dataset_valid_split"]:
                split = runner_config["dataset_valid_split"] + runner_config["dataset_train_split"]
                valid_split = int(np.floor(total_num * split))
                valid_idx = indices[train_split:valid_split]
                valid_dataset = Subset(full_dataset, valid_idx)
                valid_size = len(valid_dataset)
            else:
                valid_split = train_split

            train_idx = indices[:train_split]
            test_idx = indices[valid_split:]
            train_dataset = Subset(full_dataset, train_idx)
            test_dataset = Subset(full_dataset, test_idx)

            train_size = len(train_dataset)
            test_size = len(test_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=runner_config["batch_size"],
            shuffle=True,
            num_workers=12
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=runner_config["batch_size"],
            num_workers=12
        )

        valid_loader = None
        if valid_dataset:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=runner_config["batch_size"],
                shuffle=True,
                num_workers=12
            )
        # load test data in batches

        loaders = data_loaders(train_loader, test_loader, valid_loader)
        batch_size = runner_config["batch_size"]
        batch_first = True

        return cls(
            loaders,
            task_type=task_type,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            input_shape=input_shape,
            output_num=output_num,
            batch_size=batch_size,
            is_batch_first=batch_first
        )
