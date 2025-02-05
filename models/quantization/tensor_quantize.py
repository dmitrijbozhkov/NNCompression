import torch
from models.quantization.quant_codebook_inference import infer_codebook_model_quantization
from models.quantization.quant_codebook_perform import quantize_tensor_codebook
from models.quantization.quant_linear_inference import infer_linear_model_quantization
from models.quantization.quant_linear_perform import quantize_tensor_linear

def infer_tensor_quantize(tensor: torch.Tensor, quant_config: dict):
    """
    Infer tensor quantization based on quantization parameters

    :param tensor: Tensor to quantize
    :param quant_config: Dictionary of quantization config
    :returns: Dictionary with quantization parameters
    """
    tensor = torch.clone(tensor)

    quant_params = None
    if quant_config["quantization_type"] == "codebook":
        quant_params = infer_codebook_model_quantization(
            tensor,
            quant_config["quantization_levels"],
            quant_config["quantization_method"],
            quant_config["quantization_kmeans_init"],
            quant_config["quantization_kmeans_params"]
        )
    elif quant_config["quantization_type"] == "linear":
        quant_params = infer_linear_model_quantization(
            tensor,
            quant_config["quantization_levels"],
            quant_config["quantization_method"],
        )

    return quant_params


def quantize_tensor(tensor: torch.Tensor, quant_params: dict, device: str, quant_config: dict):
    """
    Quantize tensor using quantization config

    :param tensor: Tensor to quantize
    :param quant_config: Quzantization config to use
    :returns: Tuple of quantized tensor and quantization parameters
    """
    tensor = torch.clone(tensor)

    quant_tensor = None
    if quant_config["quantization_type"] == "codebook":
        quant_codebook = quant_params["codebook"]

        quant_codebook = quant_codebook.to(device)

        quant_tensor = quantize_tensor_codebook(
            tensor,
            quant_codebook,
            quant_config["quantization_round_strategy"]
        )
    elif quant_config["quantization_type"] == "linear":
        quant_tensor = quantize_tensor_linear(
            tensor,
            quant_params,
            quant_config["quantization_round_strategy"]
        )

    return quant_tensor
