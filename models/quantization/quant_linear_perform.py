from models.quantization.quant_linear_inference import LinearQuantParams
import torch


def _quantize_stochastic(tensor: torch.Tensor, quant_params: LinearQuantParams):
    """
    Quantize tensor using lienar quantization to lower value

    :param tensor: Tensor to quantize
    :param quant_params: Dictionary of quantization parameters
    :returns: Quantized tensor
    """
    quant_scaled = tensor / quant_params["scale"]
    quant_vals = torch.floor(quant_scaled)
    quant_up_probs = torch.abs(quant_scaled) % 1
    quant_vals = torch.bernoulli(quant_up_probs) + quant_vals
    quant = torch.clamp(
        quant_vals + quant_params["zero_point"],
        quant_params["quant_min"],
        quant_params["quant_max"]
    )

    return (quant - quant_params["zero_point"]) * quant_params["scale"]


def _quantize_down(tensor: torch.Tensor, quant_params: LinearQuantParams):
    """
    Quantize tensor using lienar quantization to lower value

    :param tensor: Tensor to quantize
    :param quant_params: Dictionary of quantization parameters
    :returns: Quantized tensor
    """
    quant = torch.clamp(
        torch.floor(tensor / quant_params["scale"]) + quant_params["zero_point"],
        quant_params["quant_min"],
        quant_params["quant_max"]
    )

    return (quant - quant_params["zero_point"]) * quant_params["scale"]


def _quantize_up(tensor: torch.Tensor, quant_params: LinearQuantParams):
    """
    Quantize tensor using lienar quantization to upper value

    :param tensor: Tensor to quantize
    :param quant_params: Dictionary of quantization parameters
    :returns: Quantized tensor
    """
    quant = torch.clamp(
        torch.ceil(tensor / quant_params["scale"]) + quant_params["zero_point"],
        quant_params["quant_min"],
        quant_params["quant_max"]
    )

    return (quant - quant_params["zero_point"]) * quant_params["scale"]


def _quantize_nearest(tensor: torch.Tensor, quant_params: LinearQuantParams):
    """
    Quantize tensor using lienar quantization to nearest value

    :param tensor: Tensor to quantize
    :param quant_params: Dictionary of quantization parameters
    :returns: Quantized tensor
    """
    quant = torch.clamp(
        torch.round(tensor / quant_params["scale"]) + quant_params["zero_point"],
        quant_params["quant_min"],
        quant_params["quant_max"]
    )

    return (quant - quant_params["zero_point"]) * quant_params["scale"]


def quantize_tensor_linear(
        tensor,
        quant_params,
        quant_strategy="nearest"
):
    """
    Quantizes tensor using linear quantization

    :param tensor: Values to quantize
    :param quant_levels: Vector of quantization levels
    :param quant_strategy: How to pick quantization level
    :returns: Quantized tensor
    """

    if quant_strategy == "nearest":
        return _quantize_nearest(tensor, quant_params)
    if quant_strategy == "up":
        return _quantize_up(tensor, quant_params)
    if quant_strategy == "down":
        return _quantize_down(tensor, quant_params)
    if quant_strategy == "stochastic":
        return _quantize_stochastic(tensor, quant_params)
