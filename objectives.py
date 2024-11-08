from hessian import hessian
import torch
import torch.nn as nn

class HessianCELoss(nn.Module):
    """
    Cross entropy with hessian
    """

    def __init__(self, net, config):
        """
        :param net: Neural network to use
        :param device: Device to put hessian results on
        :param hessian_frac: Lambda
        """
        super().__init__()
        self.net = net
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss()
        self.l1_loss = nn.MSELoss() # nn.L1Loss()

    def forward(self, x, y, perturbations=None):
        """
        :param x: logits
        :param y: targets
        """
        cross_entropy = self.cross_entropy(x, y)
        if self.config["l"] == 1.0:
            curvature = torch.tensor(0)
        else:
            # calculate first order derivative of all weights
            first_grad = torch.autograd.grad(
                cross_entropy,
                self.net.parameters(),
                create_graph=True,
                retain_graph=True
            )
            hesse = hessian(first_grad, self.net, self.device)
            curvature = torch.sum(torch.pow(hesse, 2))
        if perturbations is not None:
            perturb_loss = self.config["perturb_loss"] * self.l1_loss(x.repeat(self.config["perturb_amount"], 1), perturbations)
        else:
            perturb_loss = 0
        return (cross_entropy * self.config["l"] + curvature * (1 - self.config["l"]) + perturb_loss,
                cross_entropy.item(),
                curvature.item())
