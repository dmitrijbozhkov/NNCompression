from dataclasses import is_dataclass
from copy import deepcopy
import random
import torch
import numpy as np


class FlattenConfig:
    """Turn config into flattened dictionary"""

    def flatten(self):
        flat = deepcopy(self.__dict__)
        original_keys = list(flat.keys())

        for key in original_keys:
            if is_dataclass(flat[key]):
                flat_config = flat[key].flatten()
                for flat_key in flat_config:
                    flat[flat_key] = flat_config[flat_key]
                del flat[key]

        return flat


def assert_choices(config_value, field, choices):
    assert config_value in choices, f"Valid values for field {field} are: {choices}"


def assert_list_choices(config_value, field, choices):
    assert all(v in choices for v in config_value), f"All values in {field} must be from {choices}"


def ensure_reproducible(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
