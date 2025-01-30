from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans, MiniBatchKMeans
from models.quantization.utils import check_legal
from torch.ao.quantization import HistogramObserver
import torch


def kmeans_quant_levels(weights, level_amount, **kmeans_params):
    """
    Assign quantize centers based on KMeans algorithm

    :param weights: Model networks parameters
    :param level_amount: Amount of quantization levels
    :param kmeans_params: Dict of KMeans algorithm parameters
    :returns: torch array with quantization levels
    """
    weights = weights.to("cpu").numpy()

    kmeans = KMeans(n_clusters=level_amount, **kmeans_params)

    kmeans.fit(weights)

    quant_levels = kmeans.cluster_centers_.squeeze()

    return torch.tensor(quant_levels)


def batched_kmeans_quant_levels(weights, level_amount, **kmeans_params):
    """
    Assign quantize centers based on MiniBatchKMeans algorithm

    :param weights: Model networks parameters
    :param level_amount: Amount of quantization levels
    :param kmeans_params: Dict of KMeans algorithm parameters
    :returns: torch array with quantization levels
    """
    weights = weights.to("cpu").numpy()

    kmeans = MiniBatchKMeans(n_clusters=level_amount, **kmeans_params)

    kmeans.fit(weights)

    quant_levels = kmeans.cluster_centers_.squeeze()

    return torch.tensor(quant_levels)



def kmeans_cuda_quant_levels(weights, level_amount, device, **kmeans_params):
    """
    Quantize using parallel kmeans cuda implemetnation

    :param weights: Model networks parameters
    :param level_amount: Amount of quantization levels
    :param kmeans_params: Dict of KMeans algorithm parameters
    :param device: Device to perform KMeans on
    :returns: torch array with quantization levels
    """
    _, cluster_centers = kmeans(X=weights,
                                num_clusters=level_amount,
                                device=device,
                                **kmeans_params)

    clusters = cluster_centers.cpu().flatten()

    torch.cuda.empty_cache()

    return clusters


def uniform_affine_quantizer(weights, level_amount):
    """
    Quantize weights in a uniform affine manner

    :param weights: Tensor of weights to quantize
    :param level_amount: Amount of levels for quantization
    """
    observer = HistogramObserver(bins=level_amount)

    weights = weights.to("cpu")

    observer(weights)

    scale, zero_point = observer.calculate_qparams()

    scale = scale.type_as(weights)
    zero_point = zero_point.type_as(zero_point)

    levels = torch.arange(-(level_amount // 2), level_amount // 2)

    return (levels - zero_point) * scale


def infer_model_quantization(model, quant_levels, quant_type, quant_device, kmeans_params):
    """
    Infer quantization centers for model

    :param model: Pytorch model
    :param quant_levels: Quantization levels amount to quantize the model
    :param quant_type: Type of quantization algorithm to use
    :param quant_device: Device to perform quantization on
    :param kmeans_params: Parameters for kmeans algorithm
    """
    weights = []
    for name, module in  model.named_modules():
        if name and check_legal(module):
            params = module.weight.data.view(-1, 1).detach()
            weights.append(params)

    weights = torch.cat(weights)

    if quant_type == "kmeans":
        return kmeans_quant_levels(weights, quant_levels, **kmeans_params)
    if quant_type == "batched_kmeans":
        return batched_kmeans_quant_levels(weights, quant_levels, **kmeans_params)
    if quant_type == "kmeans_cuda":
        return kmeans_cuda_quant_levels(weights, quant_levels, quant_device, **kmeans_params)
    if quant_type == "uniform_affine":
        return uniform_affine_quantizer(weights, quant_levels)
