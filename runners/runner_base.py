from abc import ABC
from logging import Logger
from pathlib import Path
from typing import List
from datetime import datetime
from torch import nn
from torchvision.transforms import v2
from torch.optim.optimizer import Optimizer
from models.orchestrator import ModelOrchestratorBase
from study.utils import TrialConfig
from dataset.datasets_base import Dataset
import pandas as pd
import pprint
import json
import torch
import os

class RunnerBase(ABC):
    """Runner base class"""

    def __init__(self, **kwargs):
        pass

    def epoch_start(self, runner, epoch):
        pass

    def epoch_end(self, runner, epoch):
        pass

    def before_train_episode(self, runner):
        pass

    def after_train_episode(self, runner):
        pass

    def epoch_batch(self, runner, epoch, batch_idx, data, target):
        pass


class RunnerEvent(dict):

    def __init__(self, epoch_num: int, run_num: int, time: datetime):
        super().__init__(
            epoch_num=epoch_num,
            run_num=run_num,
            time=time
        )


class Runner:
    """Base class of Experiment runner"""

    # Configuration object
    config: TrialConfig

    # Common network runner properties
    net: ModelOrchestratorBase
    optimizer: Optimizer
    dataset: Dataset
    objective: nn.Module
    device: str

    # Writing data of training episodes
    result_path: Path
    checkpoint_path: Path

    # Track running stats
    running_stats: List[RunnerEvent] # Stats for each epoch
    total_epochs_trained: int
    curr_run_metadata: RunnerEvent

    # Data transforms
    transform_train: v2.Transform
    transform_test: v2.Transform

    # Plugin runners
    plugin_runners: List[RunnerBase]
    is_train_continue: bool = True
    is_epoch_batch: bool = True
    is_checkpoint: bool = False

    # Logger
    logger: Logger

    def __init__(self,
                 plugin_runners,
                 is_epoch_batch,
                 trial_id,
                 net,
                 optimizer,
                 dataset,
                 objective,
                 runner_config,
                 result_path,
                 run_num,
                 device,
                 transform_train,
                 transform_test,
                 logger,
                 **kwargs) -> None:
        # Configuration object
        self.config = runner_config
        self.trial_id = trial_id

        # Common network runner properties
        self.net = net
        self.optimizer = optimizer
        self.dataset = dataset
        self.objective = objective
        self.device = device
        self.run_num = run_num

        # Writing training state
        self.result_path = result_path
        self.checkpoint_path = self.result_path / "checkpoints" / str(run_num)

        # Running statistics
        self.running_stats = []
        self.total_epochs_trained = 0

        self.transform_train = transform_train
        self.transform_test = transform_test

        # Plugin runners
        self.plugin_runners = plugin_runners
        self.is_train_continue = True
        self.is_epoch_batch = is_epoch_batch

        self.logger = logger

    @staticmethod
    def runs_path(config, trial_id):
        """
        Get run folder path
        """
        run_folder = str(trial_id)
        return config["output"] / config["study_name"] / run_folder

    def set_checkpoint(self, checkpoint):
        """
        Set current checkpoint for model

        :param checkpoint: Checkpoint name to load
        """
        self.net.load_curr_checkpoint(checkpoint)


    def save_runs_config(self):
        """
        Save run config in run folder
        """
        runs_path = Runner.runs_path(self.config, self.trial_id)

        with open(runs_path / "config.json", "w") as f:
            conf = self.config.to_full_config()

            self.logger.info(pprint.pformat(conf))
            json.dump(conf, f, indent=2)


    def save_run_data(self, run_df):
        """
        Saves run dataframe
        """
        result_path = self.result_path / "run_data.parquet"
        self.logger.info(f"Saving run path: {result_path}")
        run_df.to_parquet(result_path)


    def train(self) -> None:
        """
        Base template for model training
        :param epochs: Number of epochs to train
        :returns: None
        """
        if self.is_train_continue:
            self.logger.info(f"Started training run: {self.run_num}")
        self.before_train_episode()

        epochs = self.config["epoch"]

        self.net.train()

        for epoch in range(1, epochs + 1):
            if not self.is_train_continue:
                break
            self.epoch_start(epoch)
            if self.is_epoch_batch:
                for batch_idx, (data, target) in enumerate(self.dataset.data_loaders.train_loader):
                    self.epoch_batch(epoch, batch_idx, data, target)
            self.total_epochs_trained += 1
            self.net.set_epochs_trained(self.total_epochs_trained)
            self.epoch_end(epoch)

        self.net.eval()

        self.after_train_episode()


    def eval_accuracy(self, dataset_split=None) -> float:
        """
        Evaluate model accuracy on dataset

        :param dataset_split: Type of dataset split to use, use test by default
        :return: float accuracy
        """
        if not dataset_split:
            dataset_split = "test"

        if dataset_split == "test":
            loader = self.dataset.data_loaders.test_loader
            data_size = self.dataset.test_size
        elif dataset_split == "valid":
            loader = self.dataset.data_loaders.valid_loader
            data_size = self.dataset.valid_size
        elif dataset_split == "train":
            loader = self.dataset.data_loaders.train_loader
            data_size = self.dataset.train_size
        else:
            raise ValueError(f"No such dataset split: {dataset_split}")

        self.net.eval()
        correct = 0
        for data, target in loader:
            data = data.to(self.device)
            data = self.transform_test(data)
            target = target.to(self.device)
            output = self.net(data).forward_out.to(self.device)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        accuracy = correct / data_size

        return accuracy


    @torch.no_grad
    def quantize(self, level_amount):
        """
        Quantize network

        :param level_amount: Amount of quantization levels to use
        :returns: Quantization centers
        """
        return self.net.quantize(level_amount)


    def run_stats_to_df(self):
        """
        Transform running stats into dataframe
        """
        return pd.DataFrame(self.running_stats)


    def epoch_start(self, epoch):
        """
        Before each epoch add RunnerEvent
        """
        self.curr_run_metadata = RunnerEvent(
            self.total_epochs_trained,
            self.run_num,
            datetime.now()
        )

        for plugin in self.plugin_runners:
            plugin.epoch_start(self, epoch)

        if self.total_epochs_trained > 0:
            self.net.delete_run_checkpoint(self.total_epochs_trained - 1)


    def epoch_end(self, epoch):
        """
        At the end of each epoch update running_stats
        """
        for plugin in self.plugin_runners:
            plugin.epoch_end(self, epoch)

        self.running_stats.append(self.curr_run_metadata)
        self.curr_run_metadata = None


    def before_train_episode(self):
        self.curr_run_metadata = RunnerEvent(
            self.total_epochs_trained,
            self.run_num,
            datetime.now()
        )

        for plugin in self.plugin_runners:
            plugin.before_train_episode(self)


    def epoch_batch(self, epoch, batch_idx, data, target):
        for plugin in self.plugin_runners:
            plugin.epoch_batch(self, epoch, batch_idx, data, target)


    def after_train_episode(self):
        for plugin in self.plugin_runners:
            plugin.after_train_episode(self)

        if self.curr_run_metadata is not None:
            self.running_stats.append(self.curr_run_metadata)
