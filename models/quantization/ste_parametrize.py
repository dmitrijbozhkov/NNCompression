from torch.autograd import Function
import torch.nn as nn
from models.quantization.tensor_quantize import quantize_tensor


class STEQuant(Function):
    """
    STE with quantization
    """

    @staticmethod
    def forward(ctx, tensor, quant_params, device, quant_config):
        """Perform quantization on weights"""
        tensor_out = quantize_tensor(tensor, quant_params, device, quant_config)
        return tensor_out


    @staticmethod
    def backward(ctx, grad_output):
        """Do straight-through for backward pass"""
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class STEParam(nn.Module):
    """Parametrization to add STE to model weights"""

    def __init__(self, device: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.quant_params = None
        self.quant_config = None


    def set_quant_params(self, quant_params: dict):
        """
        Set quantization parameters for forward

        :param quant_params: Quantization parameters dict
        """
        self.quant_params = quant_params


    def set_quant_config(self, quant_config: dict):
        """
        Set quantization config for straight through estimation

        :param quant_config: Quantization configuration
        """
        self.quant_config = quant_config


    def forward(self, weights):
        """
        Quantize weights using STE operation

        :param weights: Model weights
        :returns: Quantized weights
        """
        return STEQuant.apply(weights, self.quant_params, self.device, self.quant_config)
