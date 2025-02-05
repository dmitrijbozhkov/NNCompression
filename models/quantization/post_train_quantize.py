import torch.nn as nn
from models.quantization.quant_linear_inference import infer_linear_model_quantization
from models.quantization.quant_linear_perform import quantize_tensor_linear
from models.quantization.utils import make_weight_groups_layers, make_weight_groups_net, set_weights_net, set_weights_layers
from models.quantization.quant_codebook_inference import infer_codebook_model_quantization
from models.quantization.quant_codebook_perform import quantize_tensor_codebook


def quantize_weights(
        weight_groups: dict,
        quant_levels: int,
        device: str,
        runner_config: dict):
    """
    Quantize model using codebook quantization

    :param weight_groups: Dict of weight groups to be quantized
    :param quant_levels: Amount of quantization levels to use
    :param device: Device to put codebook to
    :param runner_config: Runner configuration with quantization parameters
    :returns: Tuple of dict with quantized groups and group centers
    """
    weight_params = {}
    weight_quantized = {}
    for w_g in weight_groups:
        if runner_config["quantization_type"] == "codebook":
            quant_params = infer_codebook_model_quantization(
                weight_groups[w_g],
                quant_levels,
                runner_config["quantization_method"],
                runner_config["quantization_kmeans_init"],
                runner_config["quantization_kmeans_params"]
            )
            weight_params[w_g] = quant_params

            quant_codebook = quant_params["codebook"]

            quant_codebook = quant_codebook.to(device)

            weight_quantized[w_g] = quantize_tensor_codebook(
                weight_groups[w_g],
                quant_codebook,
                runner_config["quantization_round_strategy"]
            )
        elif runner_config["quantization_type"] == "linear":
            quant_params = infer_linear_model_quantization(
                weight_groups[w_g],
                quant_levels,
                runner_config["quantization_method"],
            )

            weight_params[w_g] = quant_params

            weight_quantized[w_g] = quantize_tensor_linear(
                weight_groups[w_g],
                quant_params,
                runner_config["quantization_round_strategy"]
            )

    return weight_quantized, weight_params


def post_train_quantize(model: nn.Module, quant_levels: int, device: str, runner_config: dict):
    """
    Performs post-training quantization of the model based on config

    :param model: Model to quantize
    :param quant_levels: Quantization levels to use
    :param device: Device of the network
    :param runner_config: Configuration dictionary
    :returns: Tuple of quantized model and quantization group parameters
    """
    weight_groups = None
    if runner_config["quantization_granularity"] == "network":
        weight_groups = make_weight_groups_net(model)
    if runner_config["quantization_granularity"] == "layer":
        weight_groups = make_weight_groups_layers(model)

    assert weight_groups, f"No such quantization granularity: {runner_config['quantization_granularity']}"

    quantized_groups, group_params = quantize_weights(
        weight_groups,
        quant_levels,
        device,
        runner_config
    )

    if runner_config["quantization_granularity"] == "network":
        model = set_weights_net(model, quantized_groups)
    if runner_config["quantization_granularity"] == "layer":
        model = set_weights_layers(model, quantized_groups)

    return model, group_params
