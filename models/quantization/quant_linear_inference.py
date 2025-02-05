from typing import TypedDict
import torch

class LinearQuantParams(TypedDict):
    """Parameters for linear quantization"""

    zero_point: float
    scale: float
    quant_min: float
    quant_max: float


def uniform_affine_quantizer(weights, level_amount):
    """
    Quantize weights in a uniform affine manner

    :param weights: Tensor of weights to quantize
    :param level_amount: Amount of levels for quantization
    :returns: Dict with quantization parameters
    """
    w_min, w_max = torch.aminmax(weights)
    scale = (w_max - w_min) / (level_amount - 1)
    zero_point = - torch.round(w_min / scale)

    return LinearQuantParams(
        zero_point=zero_point.item(),
        scale=scale.item(),
        quant_min=0,
        quant_max=level_amount - 1
    )


def infer_linear_model_quantization(
        weights,
        quant_levels,
        quant_type) -> LinearQuantParams:
    """
    Infer linear quantization parameters for linear quantization

    :param weights: Tensor with weights
    :param quant_levels: Quantization levels amount to quantize the model
    :param kmeans_params: Parameters for kmeans algorithm
    """
    weights = weights.view(-1, 1)
    if quant_type == "uniform_affine":
        return uniform_affine_quantizer(weights, quant_levels)
