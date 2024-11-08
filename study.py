#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from optuna import Trial, create_study
from optuna.samplers import TPESampler
from copy import deepcopy
from runner import Runner
import os
import json

from utils import TrialConfig


class Study:
    """
    Performs HPO search
    """

    def __init__(self, config, search_space) -> None:
        self.search_space = search_space
        self.config = config
        self.study_path = config["output"] / config["study_name"]
        if not os.path.exists(self.study_path):
            os.makedirs(self.study_path)

        conf_path = deepcopy(config)
        conf_path["output"] = conf_path["output"].as_posix()
        with open(self.study_path / "config.json", "w") as f:
            json.dump(conf_path, f)

        if self.search_space:
            self.init_study()
        else:
            self.study = None

    def init_study(self):
        """
        Initializes optuna study
        """
        self.study = create_study(
            storage=self.config["storage"],
            study_name=self.config["study_name"],
            direction=self.config["opt_direction"],
            sampler=TPESampler(),
            load_if_exists=True
        )

    def opt_objective(self, trial = None):
        """
        Evaluate optimization objective
        """
        if trial:
            trial_id = trial._trial_id
        else:
            trial_id = None

        config = TrialConfig(self.config, self.search_space, trial)
        runner = Runner.from_config(config, trial_id)
        train_data = runner.train()
        runner.kmeans_quantize(config["quantize_levels"])
        _, _, accuracy = runner.test()
        runner.save_train_run(train_data)

        return accuracy

    def search(self):
        if self.study:
            self.study.optimize(self.opt_objective, self.config["n_trials"])
        else:
            accuracy = self.opt_objective()
            print(f"Achieved accuracy: {accuracy}")

    @staticmethod
    def prepare_config(parser: ArgumentParser):
        """
        Prepare configuration dictionary for running
        """

        # HPO Parameters
        parser.add_argument("-db", "--storage", help="Optuna storage database")
        parser.add_argument("-sn", "--study_name", type=str, required=True, help="Name of the run study")
        parser.add_argument("-od", "--opt_direction", choices=["maximize", "minimize"], help="Optimization direction")
        parser.add_argument("-nt", "--n_trials", type=int, help="Number of trials to perform")
        parser.add_argument("-ss", "--search_space", type=Path, help="Path to search space config")
        # Configuration params
        parser.add_argument("-b", "--batch_size", default=1024, type=int)
        parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
        parser.add_argument("-e", "--epoch", default=30, type=int)
        parser.add_argument("-l", "--l", default=1.0, help="lambda", type=float)
        parser.add_argument("-g", "--gpu", action='store_true', default=True)
        parser.add_argument("-wd", "--weight_decay", default=0.00001, help="Weight decay for Adam", type=float)
        parser.add_argument("-d", "--dataset", default='MNIST')
        parser.add_argument("-m", "--model", default='LeNet3_3')
        parser.add_argument("-o", "--output", type=Path, required=True, help="Output path")
        parser.add_argument("-ql", "--quantize_levels", type=int, default=8, help="Quantization levels")
        # Optimizer params
        parser.add_argument("-ot", "--optimizer_type", choices=["SGD", "Adam", "AdamW"], default="SGD", help="Optimizer to use")
        parser.add_argument("-om", "--optimizer_momentum", type=float, default=0.0, help="Optimizer momentum to use")
        parser.add_argument("-oab1", "--optimizer_adam_beta_1", type=float, default=0.9, help="Adam beta_1 parameter")
        parser.add_argument("-oab2", "--optimizer_adam_beta_2", type=float, default=0.999, help="Adam beta_2 parameter")
        parser.add_argument("-oe", "--optimizer_eps", type=float, default=1e-08, help="Optimizer epsilon value to use")
        # Scheduler params
        parser.add_argument("-st", "--scheduler_type", choices=["Lambda", "Step", "Cosine", "Cyclic", "None"], default="None", help="Scheduler type to use")
        parser.add_argument("-slm", "--scheduler_lambda_multiplier", type=float, default=1, help="Lambda scheduler multiplier")
        parser.add_argument("-sle", "--scheduler_last_epoch", type=int, default=-1, help="Lambda last epoch")
        parser.add_argument("-sss", "--scheduler_step_size", type=int, default=-1, help="Scheduler step size")
        parser.add_argument("-sg", "--scheduler_gamma", type=float, default=0.1, help="Scheduler gamma value")
        parser.add_argument("-stm", "--scheduler_T_max", type=int, default=-1, help="Scheduler T_max parameter")
        parser.add_argument("-sem", "--scheduler_eta_min", type=float, default=0.0, help="Scheduler gamma value")
        # Perturbation parameters
        parser.add_argument("-ip", "--is_perturb", action="store_true", help="Should perturbation be performed")
        parser.add_argument("-p", "--perturb_param", action="append", help="Parameter to perturb")
        parser.add_argument("-pl", "--perturb_loss", type=float, default=0.0, help="Weight for perturbation loss")
        parser.add_argument("-pm", "--perturb_mean", type=float, default=0.0, help="Mean value for perturbation noise")
        parser.add_argument("-pv", "--perturb_variance", type=float, default=0.3, help="Variance value for perturbation noise")
        parser.add_argument("-pa", "--perturb_amount", type=int, default=5, help="Amount of times to run perturbations")
        parser.add_argument("-ps", "--perturb_start", type=int, default=15, help="Epoch from which to start perturbations")

        config = parser.parse_args().__dict__

        if config.get("search_space"):
            with open(config["search_space"], "r") as f:
                search_space = json.load(f)
            config["search_space"] = config["search_space"].as_posix()
        else:
            search_space = None

        return config, search_space
