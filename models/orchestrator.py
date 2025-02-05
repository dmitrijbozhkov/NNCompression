from typing import List, NamedTuple

from joblib import Logger
from models.quantization.post_train_quantize import post_train_quantize
from models.quantization.perturb_parametrize import Perturbation
from pathlib import Path
import torch.nn as nn
import os
import torch

class ModelOutputs(NamedTuple):
    """Forward model outputs"""

    forward_out: torch.Tensor
    perturbations: List[torch.Tensor] | None


def convert_tensor(quant_params: dict):
    for p in quant_params:
        if torch.is_tensor(quant_params[p]):
            quant_params[p] = quant_params[p].detach().to("cpu").numpy()
        if isinstance(quant_params[p], dict):
            convert_tensor(quant_params[p])


class ModelOrchestratorBase(nn.Module):
    """Orchestrator that is performing model training and inference"""

    def __init__(self, net: nn.Module, runner_config: dict, result_path: Path, run_num: int, logger: Logger, device, **kwargs) -> None:
        """
        Initializes orchestrator with given model and parameters

        :param net: Model to use
        """
        super().__init__()
        self.net = net
        self.epochs_trained = 0
        self.runner_config = runner_config
        self.result_path = result_path
        self.logger = logger
        self.device = device
        self.checkpoint_path = self.result_path / "checkpoints" / str(run_num)


    def set_epochs_trained(self, epochs_trained: int):
        """
        Set amount of epochs that model was trained

        :epochs_trained: Current trained epochs amount
        """
        self.epochs_trained = epochs_trained


    def save_run_checkpoint(self, epoch: int):
        """
        Save model on epoch

        :param epoch: Epoch number to use
        """
        curr_checkpoint = str(epoch).zfill(2)
        return self.save_checkpoint(curr_checkpoint)


    def save_checkpoint(self, checkpoint_name):
        """
        Save model checkpoint on disk
        """
        checkpoint_path = self.checkpoint_path / f"{checkpoint_name}.pth"
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        torch.save(self.net.state_dict(), checkpoint_path)
        return checkpoint_name


    def delete_run_checkpoint(self, epoch: int):
        """
        Deletes checkpoint of epoch number

        :param epoch: Epoch number to delete
        """
        curr_checkpoint = str(epoch).zfill(2)
        return self.delete_checkpoint(curr_checkpoint)


    def delete_checkpoint(self, checkpoint_name, checkpoint_path=None):
        """
        Delete checkpoint from disk

        :param checkpoint_name: Name of the checkpoint to load
        :param checkpoint_path: Path to the checkpoint
        :returns: Checkpoint name to use
        """
        if not checkpoint_path:
            checkpoint_path = self.checkpoint_path / f"{checkpoint_name}.pth"
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
            self.logger.info(f"Deleted checkpoint: {checkpoint_path}")
        else:
            self.logger.info(f"Skip deleting checkpoint: {checkpoint_path}")
        return checkpoint_name


    def load_checkpoint(self, checkpoint_name, checkpoint_path=None):
        """
        Load model checkpoint

        :param checkpoint_name: Name of the checkpoint to load
        :param checkpoint_path: Path to the checkpoint to load
        :returns: Checkpoint name
        """
        if not checkpoint_path:
            checkpoint_path = self.checkpoint_path / f"{checkpoint_name}.pth"
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        self.net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        return checkpoint_name


    def forward(self, x):
        """
        Run forward for the model
        """
        output = self.net.forward(x)

        return ModelOutputs(output, None)


    @torch.no_grad
    def quantize(self, level_amount: int):
        """
        Quantize network

        :param level_amount: Amount of quantization levels to use
        :returns: Quantization centers
        """
        model, group_params = post_train_quantize(
            self.net,
            level_amount,
            self.device,
            self.runner_config
        )

        self.net = model

        convert_tensor(group_params)
        return group_params


    @staticmethod
    def get_orchestrator(model: nn.Module, result_path: Path, run_num: int, logger: Logger, runner_config: dict, device):
        """
        Creates model orchestrator from model and runner config

        :param model: Initialized model
        :param runner_config: Configuration for orchestrator
        :returns Initialized Orchestrator class
        """
        if runner_config["is_perturb_reg"]:
            return ModelOrchestratorPerturb(
                net=model,
                runner_config=runner_config,
                result_path=result_path,
                run_num=run_num,
                logger=logger,
                device=device
            )
        return ModelOrchestratorBase(
            net=model,
            runner_config=runner_config,
            result_path=result_path,
            run_num=run_num,
            logger=logger,
            device=device
        )


class ModelOrchestratorPerturb(ModelOrchestratorBase):
    """Model orchestrator for performing perturbations"""

    def __init__(self, net: nn.Module, runner_config: dict, result_path: Path, run_num: int, **kwargs) -> None:
        super().__init__(net, runner_config, result_path, run_num, **kwargs)

        self.init_perturb_reg(net, runner_config)


    def init_perturb_reg(self, net: nn.Module, runner_config: dict):
        """
        Initialize model to support perturbation regularization

        :param net: Nerual network Module
        :param runner_config: Configuration dictionary
        """
        self.perturb_parametrization = Perturbation.prepare_model_weights(
            net,
            runner_config["perturb_mean"],
            runner_config["perturb_variance"]
        )
        self.perturb_amount = runner_config["perturb_amount"]
        self.perturb_start = runner_config["perturb_start"]


    def perform_perturbation(self, data):
        """
        Perform multiple forward passes with perturbation and collect them

        :param data: Data batch
        :returns: List with perturbed network outputs or None
        """
        if self.epochs_trained < self.perturb_start:
            return None
        self.perturb_parametrization.set_is_perturb(True)

        outputs = []
        for _ in range(self.perturb_amount):
            output = self.net.forward(data)
            outputs.append(output)

        self.perturb_parametrization.set_is_perturb(False)
        return outputs


    def forward(self, x):
        """
        Run forward for the model and perform perturbation

        :param x: Model inputs
        :returns: ModelOutputs with model output and perturbations
        """
        output = self.net.forward(x)
        if self.training:
            perturbations = self.perform_perturbation(x)
        else:
            perturbations = None

        return ModelOutputs(output, perturbations)
