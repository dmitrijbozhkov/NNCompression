from typing import NamedTuple
import torch
import torch.nn as nn

from models.orchestrator import ModelOutputs


class LossHistory(NamedTuple):
    """Loss values to be saved in history"""

    loss: float # total loss
    objective_loss: float
    perturb_loss: float


class LossEvaluation(NamedTuple):
    """Evaluated loss values"""

    loss: torch.Tensor # total loss
    objective_loss: torch.Tensor
    perturb_loss: torch.Tensor

    def from_tensor(self):
        return LossHistory(
            self.loss.item(),
            self.objective_loss.item(),
            self.perturb_loss.item()
        )


class Loss(nn.Module):
    """
    Cross entropy loss with perturbation
    """

    def __init__(self, net, runner_config: dict):
        """
        :param net: Neural network to use
        :param device: Device to put hessian results on
        :param hessian_frac: Lambda
        """
        super().__init__()
        self.net = net
        self.config = runner_config
        self.cross_entropy = nn.CrossEntropyLoss()


    def forward(self, model_outputs: ModelOutputs, labels: torch.Tensor):
        """
        Objective for training classification

        :param x: logits
        :param y: targets
        :param perturbations: Perturbed outputs
        :returns: Tuple of total loss and objective loss
        """
        objective_loss = self.cross_entropy(model_outputs.forward_out, labels)

        if model_outputs.perturbations is not None:
            perturb_objectives = []
            for perturb_batch in model_outputs.perturbations:
                perturb_objective = torch.abs(self.cross_entropy(perturb_batch, labels) - objective_loss)
                perturb_objectives.append(perturb_objective.unsqueeze(0))
            perturb_objectives = torch.cat(perturb_objectives)
            perturb_loss = torch.mean(perturb_objectives)
        else:
            perturb_loss = torch.tensor(0)
        loss = (
            self.config["loss_weight"] * objective_loss +
            self.config["perturb_loss_mult"] * perturb_loss
        )
        return LossEvaluation(loss, objective_loss, perturb_loss)
