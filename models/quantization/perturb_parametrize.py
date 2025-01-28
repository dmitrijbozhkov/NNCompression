from torch import nn
from torch.nn.utils.parametrize import register_parametrization
from operator import attrgetter
from models.quantization.utils import check_legal
from torch.nn.modules.batchnorm import _NormBase
import torch


class Perturbation(nn.Module):
    """Parametrization to add perturbation noise to model weights"""

    def __init__(self, perturb_mean: float, perturb_variance: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.perturb_mean = perturb_mean
        self.perturb_variance = perturb_variance
        self.batchnorm_layers = []
        self.is_perturb = False


    def forward(self, weights):
        """
        Add perturbation to weights before applying them

        :param weights: Model weights
        :returns: Perturbed weights
        """
        if self.is_perturb:
            perturbation = torch.normal(
                self.perturb_mean,
                torch.full_like(weights, self.perturb_variance)
            )
            return weights + perturbation
        return weights


    def set_is_perturb(self, is_perturb: bool):
        """
        Switches between performing perturbation or not

        :param is_perturb: Should model weights be perturbed
        """
        self.is_perturb = is_perturb

        for b_layer in self.batchnorm_layers:
            b_layer.track_running_stats = not self.is_perturb


    def set_batchnorm_layers(self, batchnorm_layers):
        """
        Sets batchnorm layers to turn off statistics tracking

        :param batchnorm_layers: List of batchnorm layers
        """
        self.batchnorm_layers = batchnorm_layers


    @classmethod
    def prepare_model_weights(cls, model, perturb_mean: float, perturb_variance: float):
        parametrizaiton = cls(perturb_mean, perturb_variance)

        batchnorm_modules = []
        for module_name, module in  model.named_modules():
            if module_name and check_legal(module):
                register_parametrization(module, "weight", parametrizaiton)
            if isinstance(module, _NormBase):
                batchnorm_modules.append(module)

        parametrizaiton.set_batchnorm_layers(batchnorm_modules)

        return parametrizaiton
