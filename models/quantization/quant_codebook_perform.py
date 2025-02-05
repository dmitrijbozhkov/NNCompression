import torch


def _quantize(tensor, quant_levels, quant_idx):
    """
    Quantize tensor with appropriate indices to quantization centers
    :param tensor: tensor to quantize
    :param quant_levels: Quantization levels to compress to
    :param quant_idx: Indices fo levels for each value
    :returns: Quantized tensor inplace
    """
    tensor.copy_(quant_levels[quant_idx].view(tensor.shape))
    return tensor


def _quantize_nearest(tensor, quant_matrix, quant_levels):
    """
    Quantizes tensor values to nearest

    :param tensor: Values to quantize
    :param quant_matrix: Matrix of differences between tensor values and quant levels
    :param quant_levels: Quantization values to use
    :returns: Quantized tensor
    """
    quant_idx = quant_matrix.abs().argmin(axis=1)
    return _quantize(tensor, quant_levels, quant_idx)

def _quantize_up(tensor, quant_matrix, quant_levels):
    """
    Quantizes tensor values up

    :param tensor: Values to quantize
    :param quant_matrix: Matrix of differences between tensor values and quant levels
    :param quant_levels: Quantization values to use
    :returns: Quantized tensor
    """
    max_level_mask = (tensor >= quant_levels.max()).view(-1)
    quant_matrix[quant_matrix > 0] = 99999999
    quant_idx = quant_matrix.abs().argmin(axis=1)
    quant_idx[max_level_mask] = len(quant_levels) - 1
    return _quantize(tensor, quant_levels, quant_idx)


def _quantize_down(tensor, quant_matrix, quant_levels):
    """
    Quantizes tensor values down

    :param tensor: Values to quantize
    :param quant_matrix: Matrix of differences between tensor values and quant levels
    :param quant_levels: Quantization values to use
    :returns: Quantized tensor
    """
    min_level_mask = (tensor <= quant_levels.min()).view(-1)
    quant_matrix[quant_matrix < 0] = 99999999
    quant_idx = quant_matrix.abs().argmin(axis=1)
    quant_idx[min_level_mask] = 0
    return _quantize(tensor, quant_levels, quant_idx)


def _quantize_stochastic(tensor, quant_matrix, quant_levels):
    """
    Perform stochastic quantization where quantized value is chosen based on distance between value and quantization level

    :param tensor: Tensor of values to quantize
    :param quant_matrix: Quantization matrix of distances
    :param quant_levels: Quantization levels array
    :returns: Quantized tensor
    """
    min_level_mask = (tensor <= quant_levels.min()).view(-1)
    max_level_mask = (tensor >= quant_levels.max()).view(-1)

    closest_1 = quant_matrix.abs().argmin(axis=1)
    closest_1_dist = quant_matrix[torch.arange(15), closest_1].abs()
    quant_matrix[torch.arange(15), closest_1] = 9999999

    closest_2 = quant_matrix.abs().argmin(axis=1)
    closest_2_dist = quant_matrix[torch.arange(15), closest_2].abs()

    closest_1_probs = closest_1_dist / (closest_1_dist + closest_2_dist)
    draws = torch.bernoulli(closest_1_probs).bool()

    quant_idx = torch.where(draws, closest_1, closest_2)
    quant_idx[max_level_mask] = len(quant_levels) - 1
    quant_idx[min_level_mask] = 0

    return _quantize(tensor, quant_levels, quant_idx)


def quantize_tensor_codebook(
        tensor,
        quant_levels,
        quant_strategy="nearest",
):
    """
    Quantizes tensor using levels and quantization strategy

    :param tensor: Values to quantize
    :param quant_levels: Vector of quantization levels
    :param quant_strategy: How to pick quantization level
    :returns: Quantized tensor
    """
    tensor_rows = tensor.view(-1, 1)
    quant_columns = quant_levels.view(1, -1)

    quant_matrix = tensor_rows - quant_columns

    if quant_strategy == "nearest":
        return _quantize_nearest(tensor, quant_matrix, quant_levels)
    if quant_strategy == "up":
        return _quantize_up(tensor, quant_matrix, quant_levels)
    if quant_strategy == "down":
        return _quantize_down(tensor, quant_matrix, quant_levels)
    if quant_strategy == "stochastic":
        return _quantize_stochastic(tensor, quant_matrix, quant_levels)
