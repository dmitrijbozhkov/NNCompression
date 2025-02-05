from torch.autograd import Function
import torch
import torch.nn as nn


class Freeze(Function):
    """
    Operation for freezing some weights for backward pass
    """

    @staticmethod
    def forward(ctx, x, freeze_matrix):
        ctx.save_for_backward(freeze_matrix)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        freeze_matrix = ctx.saved_tensors
        grad_input = torch.clone(grad_output)
        grad_input[freeze_matrix] = 0
        return grad_input, None


class FreezeParam(nn.Module):
    """Prevents some paramters from being changed by backward pass"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.freeze_matrix = None


    def set_freeze_matrix(self, freeze_matrix):
        """
        Set binary matrix to freeze weights

        :param freeze_matrix: Binary matrix for freezing weights
        """
        self.freeze_matrix = freeze_matrix


    def forward(self, weights):
        """
        Freeze weights using freeze matrix

        :param weights: Weights to apply freezing to
        :returns: Weights with freezing operation
        """
        if self.freeze_matrix is not None:
            return Freeze.apply(weights, self.freeze_matrix)
        return weights
