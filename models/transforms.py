from torchvision.transforms import v2
import torch


def get_dtype(dtype_str: str):
    """
    Returns pytorch dtype from string
    :param dtype_str: pytorch dtype as string
    :returns: pytorch.dtype
    """
    return getattr(torch, dtype_str)


def canonical_initialize(runner_config: dict):
    if runner_config["dataset"] == "cifar100":
        mean = [x / 255.0 for x in [0.507, 0.487, 0.441]]
        std = [x / 255.0 for x in [0.267, 0.256, 0.276]]
        transform_train = v2.Compose([
            v2.RandomCrop(32, 4),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(15),
            v2.Normalize(mean=mean, std=std)
        ])
        transform_test = v2.Compose([
            v2.Normalize(mean=mean, std=std)
        ])

    return transform_train, transform_test


def init_transform(transform_dict: dict):
    """
    Initialize transform from configuration dictionary

    :param transform_dict: Transformation dictionary
    :returns: Initialized transform
    """
    if transform_dict["type"] == v2.ToImage.__name__:
        return v2.ToImage()
    if transform_dict["type"] == v2.ToDtype.__name__:
        return v2.ToDtype(
            dtype=get_dtype(transform_dict["dtype"]),
            scale=transform_dict.get("scale", True)
        )
    if transform_dict["type"] == v2.RandomCrop.__name__:
        return v2.RandomCrop(
            size=transform_dict["size"],
            padding=transform_dict["padding"],
        )
    if transform_dict["type"] == v2.RandomHorizontalFlip.__name__:
        return v2.RandomHorizontalFlip(
            p=transform_dict.get("p", 0.5)
        )
    if transform_dict["type"] == v2.RandomRotation.__name__:
        return v2.RandomRotation(
            degrees=transform_dict["degrees"]
        )
    if transform_dict["type"] == v2.Normalize.__name__:
        return v2.Normalize(
            mean=transform_dict["mean"],
            std=transform_dict["std"]
        )


def get_transform(runner_config: dict):
    """
    Initializes transforms for training

    :param runner_config: Configuration for runner
    :returns: Tuple of transforms for training and testing
    """
    if runner_config["is_canonical_transforms"]:
        return canonical_initialize(runner_config)

    transform_train = []
    for t_config in runner_config["training_transforms"]:
        transform = init_transform(t_config)
        transform_train.append(transform)

    transform_test = []
    for t_config in runner_config["testing_transforms"]:
        transform = init_transform(t_config)
        transform_test.append(transform)

    return v2.Compose(transform_train), v2.Compose(transform_test)
