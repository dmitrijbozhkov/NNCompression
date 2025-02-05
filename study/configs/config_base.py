from logging import Logger
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, TypedDict
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.transforms import v2
from torch import nn
from dataset.datasets_base import Dataset
from models.models_base import get_model
from models.optimizer import get_optimizer, get_scheduler
from models.orchestrator import ModelOrchestratorBase
from models.transforms import get_transform
from objectives import Loss
from runners.runner_base import Runner
from study.configs.config_perturb_regularization import PerturbRegularzationConfig
from study.configs.common import assert_choices, FlattenConfig, assert_list_choices, ensure_reproducible
from study.utils import TrialConfig, get_device
import os
import shutil
import logging

class RunnerConfigDict(TypedDict):
    """Dictionary of all initialized objects for initializing the runner"""

    trial_id: str
    net: nn.Module
    optimizer: Optimizer
    dataset: Dataset
    scheduler: None | LRScheduler
    objective: nn.Module
    runner_config: TrialConfig
    result_path: Path
    run_num: int
    device: str
    transform_train: v2.Transform
    transform_test: v2.Transform
    logger: Logger


@dataclass
class HPOConfig(FlattenConfig):
    """Configuration for HPO"""
    database: str = ""
    opt_direction: str = "minimize"
    hp_sampler: str = "tpe"
    n_trials: int = 0
    optimization_objective: str = "test_accuracy"

    search_space: dict = field(default_factory=dict)
    is_hpo: bool = False

    def __post_init__(self):
        assert_choices(self.opt_direction, "opt_direction", ["maximize", "minimize"])
        assert_choices(self.hp_sampler, "hp_sampler", ["grid", "tpe"])
        # assert_choices(self.optimization_objective, "optimization_objective", ["accuracy"])


@dataclass
class TrainingConfig(FlattenConfig):
    """Configuration for training"""
    # Common
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    epoch: int = 100
    gpu: str = ""
    dataset: str = "MNIST"
    is_dataset_canon_split: bool = False
    dataset_train_split: float = 0.6
    dataset_valid_split: float = 0.2
    dataset_test_split: float = 0.2
    model: str = "LeNet3_3"
    loss_weight: float = 1.0

    # Optimizer configurations
    optimizer_type: str = "sgd"
    optimizer_momentum: float = 0.0
    optimizer_adam_beta_1: float = 0.9
    optimizer_adam_beta_2: float = 0.999
    optimizer_eps: float = 1e-08
    optimizer_spsa_initial_lr: float = 0.1
    optimizer_spsa_initial_perturb_magnitude: float = 0.01
    optimizer_spsa_lr_stabilizer: float | None =  None
    optimizer_spsa_lr_decay: float = 0.602
    optimizer_spsa_perturb_decay: float = 0.101

    # Scheduler configurations
    scheduler_type: str = "const"
    scheduler_step_type: str = "epoch"
    scheduler_lambda_multiplier: float = 1
    scheduler_lambda_last_epoch: int = -1
    scheduler_step_size: int = -1
    scheduler_gamma: float = 0.1
    scheduler_T_max: int = -1
    scheduler_eta_min: float = 0.0
    scheduler_milestones: List[int] = field(default_factory=list)

    # Evaluation
    evaluate_step: int = 1
    evaluate_metrics: List[str] = field(default_factory=lambda : ["accuracy"])
    evaluate_datasets: List[str] = field(default_factory=lambda : ["test"])

    # Early stopping
    is_early_stopping: bool = False
    early_stopping_min_delta: float = 0
    early_stopping_tolerance: int = 0
    early_stopping_eval_metric: str = ""

    is_canonical_transforms: bool = True
    training_transforms: List[Dict] = field(default_factory=list)
    testing_transforms: List[Dict] = field(default_factory=list)

    output: Path = field(default_factory=Path)
    is_collect_objective_values: bool = False

    def __post_init__(self):
        assert_choices(self.optimizer_type, "optimizer_type", [
            "sgd",
            "adam",
            "adamw",
            "spsa"
        ])
        assert_choices(self.scheduler_type, "scheduler_type", [
            "lambda",
            "step",
            "cosine",
            "cyclic",
            "const",
            "multistep"
        ])
        assert_choices(self.scheduler_step_type, "scheduler_step_type", [
            "epoch",
            "batch"
        ])
        assert_list_choices(self.evaluate_metrics, "evaluate_metrics", [
            "accuracy"
        ])
        assert_list_choices(self.evaluate_datasets, "evaluate_datasets", [
            "train",
            "valid",
            "test"
        ])

        self.output = Path(self.output)

@dataclass
class QuantizationConfig(FlattenConfig):
    """Configuration for quantization"""
    is_quantize: bool = False
    quantization_type: str = "codebook"
    quantization_method: str = "batched_kmeans"
    quantization_kmeans_init: str = "uniform"
    quantization_granularity: str = "network"
    quantization_kmeans_params: dict = field(default_factory=dict)
    quantization_round_strategy: str = "nearest"
    quantize_levels: List[int] = field(default_factory=lambda: [4])

    def __post_init__(self):
        assert_choices(self.quantization_granularity, "quantization_granularity", [
            "network",
            "layer",
        ])
        assert_choices(self.quantization_type, "quantization_type", [
            "linear",
            "codebook"
        ])
        assert_choices(self.quantization_method, "quantization_method", [
            "kmeans",
            "batched_kmeans",
            "uniform_affine"
        ])
        assert_choices(self.quantization_round_strategy, "quantization_round_strategy", [
            "nearest",
            "up",
            "down",
            "stochastic"
        ])
        assert_choices(self.quantization_kmeans_init, "quantization_kmeans_init", [
            "k-means++",
            "random",
            "uniform"
        ])

@dataclass
class StudyConfig(FlattenConfig):
    """Global config for study"""

    study_name: str
    seed: int = 0
    reruns: int = 0
    epochs_test: int = -1
    logging_level: str = "info"
    is_record_loss: bool = False
    load_checkpoint: Path = field(default_factory=Path)

    hpo_config: HPOConfig = field(default_factory=dict)
    training_config: TrainingConfig = field(default_factory=dict)
    quantization_config: QuantizationConfig = field(default_factory=dict)
    perturb_regularization_config: PerturbRegularzationConfig = field(default_factory=dict)

    def __post_init__(self):
        self.hpo_config = HPOConfig(**self.hpo_config)
        self.training_config = TrainingConfig(**self.training_config)
        self.quantization_config = QuantizationConfig(**self.quantization_config)
        self.perturb_regularization_config = PerturbRegularzationConfig(**self.perturb_regularization_config)
        self.load_checkpoint = Path(self.load_checkpoint)


    def make_runner_config_dict(self, run_num=0, trial=None, trial_id=None) -> RunnerConfigDict:
        """
        Make runner config dictionary to feed into runner
        """
        runner_config = self.flatten()

        ensure_reproducible(self.seed)

        runner_config = TrialConfig(runner_config, self.hpo_config.search_space, trial)

        logging_level = logging.NOTSET
        if self.logging_level == "debug":
            logging_level = logging.DEBUG
        elif self.logging_level == "info":
            logging_level = logging.INFO
        elif self.logging_level == "warning":
            logging_level = logging.WARNING
        elif self.logging_level == "error":
            logging_level = logging.ERROR
        elif self.logging_level == "critical":
            logging_level = logging.CRITICAL

        logging.basicConfig(level=logging_level)

        logger = logging.getLogger("study_logger")


        if not trial_id:
            trial_id = runner_config.config_to_id()
            logger.info(f"Trial not set, trial id is: {trial_id}")

        result_path = Runner.runs_path(runner_config, trial_id)
        run_path = result_path / "checkpoints" / str(run_num)
        if os.path.exists(run_path):
            shutil.rmtree(run_path)
        os.makedirs(run_path)

        device = get_device(runner_config)
        dataset = Dataset.get_dataset(runner_config)
        model = get_model(dataset, runner_config)

        model.to(device)

        orchestrator = ModelOrchestratorBase.get_orchestrator(
            model,
            result_path,
            run_num,
            logger,
            runner_config,
            device
        )

        if runner_config["load_checkpoint"]:
            orchestrator.load_checkpoint("", runner_config["load_checkpoint"])

        train_transforms, test_transforms = get_transform(runner_config)

        if runner_config["optimizer_type"] == "spsa":
            objective = nn.CrossEntropyLoss(reduction="none")
        else:
            objective = Loss(model, runner_config)
        optimizer = get_optimizer(
            model,
            objective,
            device,
            runner_config
        )
        scheduler = get_scheduler(optimizer, runner_config)

        return RunnerConfigDict(
            trial_id=trial_id,
            net=orchestrator,
            optimizer=optimizer,
            dataset=dataset,
            scheduler=scheduler,
            objective=objective,
            runner_config=runner_config,
            result_path=result_path,
            run_num=run_num,
            device=device,
            transform_train=train_transforms,
            transform_test=test_transforms,
            logger=logger
        )
