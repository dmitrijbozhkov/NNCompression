import torch.nn as nn
import torch

LEGAL_MODULES = [
    nn.Conv2d,
    nn.Linear
]

def check_legal(module):
    """
    Check if module is legal to quantize
    """
    for module_type in LEGAL_MODULES:
        if isinstance(module, module_type):
            return True
    return False


def make_weight_groups_layers(model):
    """
    Get weight groups to quantize in a model. Returns a group for each legal layer

    :param model: Model to get weight groups from
    :returns: Dictionary of weight groups
    """
    weight_groups = {}
    for name, module in model.named_modules():
        if name and check_legal(module):
            weight_groups[name] = module.weight.view(-1).detach()

    return weight_groups

def make_weight_groups_net(model):
    """
    Get weight groups to quantize in a model. Returns a single group for whole network

    :param model: Model to be quantized
    :returns: Dictionary with single model parameter array
    """
    weights = []
    for name, module in  model.named_modules():
        if name and check_legal(module):
            params = module.weight.data.view(-1, 1).detach()
            weights.append(params)

    weights = torch.cat(weights)

    return {
        "model": weights
    }

def set_weights_layers(model, weight_dict):
    """
    Set weight dictionary layer-vise

    :param model: Model to set the weights to
    :param weight_dict: Dictionary with weights to set the model to
    :returns: model to set quantized values to
    """
    for name, module in model.named_modules():
        weight_change = weight_dict.get(name)
        if weight_change is not None:
            tensor_set = torch.reshape(weight_change, module.weight.shape)
            module.weight.data.copy_(tensor_set)

    return model

def set_weights_net(model, weight_dict):
    """
    Set weight dictionary for whole network from one network tensor

    :param model: Model to set weights to
    :param weight_dict: Dictionary of model weights
    :returns: Model with set weights
    """
    pointer = 0
    total_tensor = weight_dict["model"]
    for name, module in model.named_modules():
        if name and check_legal(module):
            num_weight_elems = module.weight.numel()
            tensor_set = total_tensor[pointer:num_weight_elems + pointer]
            tensor_set = torch.reshape(tensor_set, module.weight.shape)
            module.weight.data.copy_(tensor_set)
            pointer += num_weight_elems

    return model
