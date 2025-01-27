from logging import Logger
from torch import nn
from models.quantization.perturb_parametrize import Perturbation
from models.spsa import SPSA
from runners.runner_base import Runner, RunnerBase


class RunnerTrain(RunnerBase):
    """Runner plugin for default model training"""


    def __init__(self, net: nn.Module, runner_config: dict, logger: Logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self.epoch_train_loss = []
        self.is_record_loss = runner_config["is_record_loss"]

        self.is_perturb_reg = runner_config["is_perturb_reg"]
        if self.is_perturb_reg:
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


    def epoch_start(self, runner, epoch):
        """
        Set model to train mode and reset loss accumulation
        """
        runner.net.train()
        self.epoch_train_loss = []


    def perform_perturbation(self, runner: Runner, data):
        """
        Perform multiple forward passes with perturbation and collect them

        :param data: Data batch
        :returns: List with perturbed network outputs or None
        """
        if runner.total_epochs_trained < self.perturb_start:
            return None
        self.perturb_parametrization.set_is_perturb(True)

        outputs = []
        for _ in range(self.perturb_amount):
            output = runner.net.forward(data)
            outputs.append(output)

        self.perturb_parametrization.set_is_perturb(False)
        return outputs


    def epoch_batch(self, runner: Runner, epoch, batch_idx, data, target):
        """
        Perform training on batch
        """
        data = data.to(runner.device)
        target = target.to(runner.device)
        runner.optimizer.zero_grad()
        data = runner.transform_train(data)
        output = runner.net.forward(data)
        perturbations = self.perform_perturbation(runner, data) if self.is_perturb_reg else None
        objective = runner.objective(output, target, perturbations)
        objective.loss.backward()
        runner.optimizer.step()
        if self.is_record_loss:
            self.epoch_train_loss.append(objective.from_tensor())


    def epoch_end(self, runner, epoch):
        """
        Set model to eval mode and save accumulated losses
        """
        runner.net.eval()
        if self.is_record_loss:
            runner.curr_run_metadata["train_loss"] = self.epoch_train_loss
            self.logger.info(f"Epoch {epoch} train loss: {self.epoch_train_loss}")


class RunnerSchedule(RunnerTrain):
    """Runner plugin for scheduled model training"""

    def __init__(self, scheduler, runner_config, **kwargs):
        super().__init__(runner_config=runner_config, scheduler=scheduler, **kwargs)
        self.scheduler = scheduler
        self.scheduler_step_type = runner_config["scheduler_step_type"]

    def epoch_batch(self, runner: Runner, epoch, batch_idx, data, target):
        super().epoch_batch(runner, epoch, batch_idx, data, target)
        if self.scheduler_step_type == "batch":
            self.scheduler.step()

    def epoch_end(self, runner, epoch):
        super().epoch_end(runner, epoch)
        if self.scheduler_step_type == "epoch":
            self.scheduler.step()


class RunnerTrainEpoch(RunnerTrain):
    """Training runner for training without batch data"""

    def __init__(self, net: nn.Module, runner_config: dict, logger: Logger, optimizer, **kwargs):
        super().__init__(net, runner_config, logger, **kwargs)
        self.optimizer = optimizer


    def epoch_start(self, runner, epoch):
        """Perform optimizer step on epoch start"""
        self.optimizer.step(runner.dataset.data_loaders.train_loader)


    def epoch_batch(self, runner: Runner, epoch, batch_idx, data, target):
        """
        Ignore batch training
        """
        pass
