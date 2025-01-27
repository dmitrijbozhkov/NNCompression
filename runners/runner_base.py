from abc import ABC
from logging import Logger
from pathlib import Path
from typing import List
from datetime import datetime
from torch import nn
from torch.optim.optimizer import Optimizer
from models.quantization.quant_inference import infer_model_quantization
from models.quantization.quant_perform import quantize_model_weights
from study.utils import TrialConfig
from dataset.datasets_base import Dataset
from torchvision import transforms
import pandas as pd
import pprint
import json
import torch

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
    net: nn.Module
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

    # Plugin runners
    plugin_runners: List[RunnerBase]
    is_train_continue: bool = True
    is_epoch_batch: bool = True

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

        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]
        # mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        # std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [x / 255.0 for x in[0.507, 0.487, 0.441]]
        std = [x / 255.0 for x in [0.267, 0.256, 0.276]]

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.transform_test = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])

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

    def save_checkpoint(self, checkpoint_name):
        """
        Save model checkpoint on disk
        """
        checkpoint_path = self.checkpoint_path / f"{checkpoint_name}.pth"
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        torch.save(self.net.state_dict(), checkpoint_path)


    def load_checkpoint(self, checkpoint_name):
        """
        Load model checkpoint
        """
        checkpoint_path = self.checkpoint_path / f"{checkpoint_name}.pth"
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        self.net.load_state_dict(torch.load(checkpoint_path, weights_only=True))


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
            output = self.net(data).to(self.device)
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
        quant_centers = infer_model_quantization(
            self.net,
            level_amount,
            self.config["quantization_type"],
            self.config["quantization_device"],
            self.config["quantization_kmeans_params"]
        )
        quant_centers = quant_centers.to(self.device)

        quantize_model_weights(
            self.net,
            quant_centers,
            self.config["quantization_strategy"]
        )

        return quant_centers.cpu().numpy()

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

    def epoch_end(self, epoch):
        """
        At the end of each epoch update running_stats
        """
        for plugin in self.plugin_runners:
            plugin.epoch_end(self, epoch)

        self.running_stats.append(self.curr_run_metadata)

    def before_train_episode(self):
        for plugin in self.plugin_runners:
            plugin.before_train_episode(self)

    def epoch_batch(self, epoch, batch_idx, data, target):
        for plugin in self.plugin_runners:
            plugin.epoch_batch(self, epoch, batch_idx, data, target)

    def after_train_episode(self):
        for plugin in self.plugin_runners:
            plugin.after_train_episode(self)
