from dataclasses import dataclass
from study.configs.common import FlattenConfig


@dataclass
class PerturbRegularzationConfig(FlattenConfig):
    """Configuration for perturbation regularization"""
    is_perturb_reg: bool = False
    perturb_loss_mult: float = 0.0
    perturb_mean: float = 0.0
    perturb_variance: float = 0.3
    perturb_amount: int = 5
    perturb_start: int = 0
