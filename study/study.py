#!/usr/bin/env python3
from optuna import create_study
from optuna.samplers import TPESampler, GridSampler
from omegaconf import DictConfig, OmegaConf

import torch
import pandas as pd
import os
import hydra
from runners.runner_factory import create_runner
from study.configs.config_base import StudyConfig
from visualization.runner_output_utils import preprocess_run_data


class Study:
    """
    Study class to initialize model runs
    """

    def __init__(self, study_config: StudyConfig) -> None:
        self.study_config = study_config
        self.search_space = study_config.hpo_config.search_space

        self.study_path = self.study_config.training_config.output / self.study_config.study_name

        if not os.path.exists(self.study_path):
            os.makedirs(self.study_path)

        if self.study_config.hpo_config.is_hpo:
            self.init_study()
        else:
            self.study = None

    def init_study(self):
        """
        Initializes optuna study
        """
        if self.study_config.hpo_config.hp_sampler == "tpe":
            sampler = TPESampler()
        elif self.study_config.hpo_config.hp_sampler == "grid":
            sampler = GridSampler(self.search_space)

        self.study = create_study(
            storage=self.study_config.hpo_config.database,
            study_name=self.study_config.study_name,
            direction=self.study_config.hpo_config.opt_direction,
            sampler=sampler,
            load_if_exists=True
        )


    def opt_objective(self, trial = None):
        """
        Evaluate optimization objective
        """
        total_runs = 1 + self.study_config.reruns

        trial_id = str(trial.number) if trial else None

        run_data = []
        for run_num in range(total_runs):
            runner_config = self.study_config.make_runner_config_dict(run_num, trial, trial_id)
            runner = create_runner(runner_config)
            runner.train()
            runner.save_runs_config()
            run_df = runner.run_stats_to_df()
            run_data.append(run_df)
            if run_num == total_runs - 1:
                run_data = pd.concat(run_data)
                runner.save_run_data(run_data)
                del runner
                torch.cuda.empty_cache()

        mean_df, std_df, selected_vals = preprocess_run_data(run_data)

        if self.study_config.hpo_config.is_hpo:
            return selected_vals[self.study_config.hpo_config.optimization_objective]
        else:
            return selected_vals


    def start(self):
        if self.study:
            self.study.optimize(self.opt_objective, self.study_config.hpo_config.n_trials)
        else:
            accuracy = self.opt_objective()
            print(f"Achieved accuracy: {accuracy}")


    def load_checkpoint(self, trial_id, run_num, checkpoint):
        """
        Load trained model checkpoint

        :param trial_id: trial_id fodler name that contains the study
        :param run_num: # of run to load
        :param checkpoint: checkpoint number to load
        :returns: loaded runner
        """
        runner_config = self.study_config.make_runner_config_dict(run_num, None, trial_id)
        runner = create_runner(runner_config, checkpoint)
        return runner


    @hydra.main(version_base=None, config_path="../config", config_name="default")
    @staticmethod
    def study_main(cfg: DictConfig):
        """
        Main method for initializing config from terminal

        :param cfg: hydra configuration
        :returns: StudyConfig
        """
        config = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

        study_config = StudyConfig(**config)

        study = Study(study_config)

        print(study.start())


    @staticmethod
    def study_initialize(config_name="template", overrides=None):
        """
        Method for initializing config programmatically

        :param overrides: Configuration parameters to override like in terminal
        :returns: StudyConfig
        """
        if not overrides:
            overrides = []
        with hydra.initialize(version_base=None, config_path="../config"):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            config = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
            return StudyConfig(**config)
