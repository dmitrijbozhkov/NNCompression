from typing import TypedDict
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import torch

class CodebookQuantParams(TypedDict):
    """Parameters for codebook quantization"""

    codebook: torch.Tensor


def initialize_kmeans(weights, level_amount):
    """
    Initialize K-means grid uniformly

    :param weights: Weights to quantize
    :param level_amount: Amount of levels
    :returns:
    """
    w_max = weights.max()
    w_min = weights.min()
    init_levels = np.linspace(w_min, w_max, level_amount)
    return init_levels[:, np.newaxis]


def kmeans_quant_levels(weights, level_amount, init_quant, **kmeans_params):
    """
    Assign quantize centers based on KMeans algorithm

    :param weights: Model networks parameters
    :param level_amount: Amount of quantization levels
    :param kmeans_params: Dict of KMeans algorithm parameters
    :returns: torch array with quantization levels
    """
    weights = weights.view(-1, 1)

    weights = weights.to("cpu").numpy()

    if init_quant == "uniform":
        init_levels = initialize_kmeans(weights, level_amount)
    else:
        init_levels = init_quant

    kmeans = KMeans(
        n_clusters=level_amount,
        init=init_levels
        **kmeans_params)

    kmeans.fit(weights)

    quant_levels = kmeans.cluster_centers_.squeeze()

    return torch.tensor(quant_levels)


def batched_kmeans_quant_levels(weights, level_amount, init_quant, **kmeans_params):
    """
    Assign quantize centers based on MiniBatchKMeans algorithm

    :param weights: Model networks parameters
    :param level_amount: Amount of quantization levels
    :param kmeans_params: Dict of KMeans algorithm parameters
    :returns: torch array with quantization levels
    """
    weights = weights.view(-1, 1)

    weights = weights.to("cpu").numpy()

    if init_quant == "uniform":
        init_levels = initialize_kmeans(weights, level_amount)
    else:
        init_levels = init_quant

    kmeans = MiniBatchKMeans(
        n_clusters=level_amount,
        init=init_levels,
        **kmeans_params
    )

    kmeans.fit(weights)

    quant_levels = kmeans.cluster_centers_.squeeze()

    return torch.tensor(quant_levels)


def histogram_quantizer(weights, level_amount):
    """
    Quantize weights using a histogram

    :param weights: Tensor of weights to quantize
    :param level_amount: Amount of levels for quantization
    :returns: Tensor with codebook
    """
    return torch.histogram(weights, level_amount - 1).bin_edges


def quantile_quantizer(weights, level_amount):
    """
    Quantize weights using quantiles

    :param weights: Tensor of weights to quantize
    :param level_amount: Amount of levels for quantization
    :returns: Tensor with codebook
    """
    quantiles = torch.arange(0, level_amount) / (level_amount - 1)
    return torch.quantile(weights, quantiles)


def infer_codebook_model_quantization(
        weights,
        quant_levels,
        quant_type,
        quant_kmeans_init,
        kmeans_params):
    """
    Infer quantization centers for model

    :param model: Pytorch model
    :param quant_levels: Quantization levels amount to quantize the model
    :param quant_type: Type of quantization algorithm to use
    :param quant_device: Device to perform quantization on
    :param kmeans_params: Parameters for kmeans algorithm
    """
    codebook = None
    if quant_type == "kmeans":
        codebook = kmeans_quant_levels(weights, quant_levels, quant_kmeans_init, **kmeans_params)
    if quant_type == "batched_kmeans":
        codebook = batched_kmeans_quant_levels(weights, quant_levels, quant_kmeans_init, **kmeans_params)
    if quant_type == "histogram":
        codebook = histogram_quantizer(weights, quant_levels)
    if quant_type == "quantile":
        codebook = quantile_quantizer(weights, quant_levels)

    assert codebook is not None, f"No such quantization type: {quant_type}"

    return CodebookQuantParams(
        codebook=codebook
    )
