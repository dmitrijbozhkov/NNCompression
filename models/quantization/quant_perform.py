from models.quantization.utils import check_legal


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
    quant_idx = quant_matrix.abs().min(axis=1).indices
    return _quantize(tensor, quant_levels, quant_idx)

def _quantize_up(tensor, quant_matrix, quant_levels):
    """
    Quantizes tensor values up

    :param tensor: Values to quantize
    :param quant_matrix: Matrix of differences between tensor values and quant levels
    :param quant_levels: Quantization values to use
    :returns: Quantized tensor
    """
    quant_matrix[quant_matrix > 0] = 99999999
    quant_idx = quant_matrix.abs().min(axis=1).indices
    return _quantize(tensor, quant_levels, quant_idx)


def _quantize_down(tensor, quant_matrix, quant_levels):
    """
    Quantizes tensor values down

    :param tensor: Values to quantize
    :param quant_matrix: Matrix of differences between tensor values and quant levels
    :param quant_levels: Quantization values to use
    :returns: Quantized tensor
    """
    quant_matrix[quant_matrix < 0] = 99999999
    quant_idx = quant_matrix.abs().min(axis=1).indices
    return _quantize(tensor, quant_levels, quant_idx)


def quantize_tensor(
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


def quantize_model_weights(model, quant_levels, quant_strategy):
    """
    Quantizes model weight parameters using quantization levels and strategy

    :param model: Pytorch model to quantize
    :param quant_levels: Vector of quantization levels
    :param quant_strategy: How to pick quantization level
    """
    for name, module in  model.named_modules():
        if name and check_legal(module):
            quantize_tensor(module.weight, quant_levels, quant_strategy)
