#!/usr/bin/env python3
from optuna import create_study
from optuna.samplers import TPESampler, GridSampler
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.tensorboard.writer import SummaryWriter
import pandas as pd
import os
import hydra

# from utils import TrialConfig, extract_model_weights, get_mean_std_run
from runners.runner_factory import create_runner
from visualize import plot_mean_run_acc, plot_quant_acc, plot_quant_train_acc, plot_run_loss, plot_test_acc, plot_mean_run_loss, plot_weights
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



    def make_summary_writer(self, config, trial_id):
        return SummaryWriter(config["run_summary"] / str(trial_id))


    def write_study_summary(self, config, writer, trial, run_df, mean_df, std_df, selected_vals):
        # Write run data
        # writer.add_hparams(config.to_valid_hp(), selected_vals, run_name=trial)

        run_loss_figure = plot_run_loss(run_df, config["epoch"])

        writer.add_figure("Run Loss", run_loss_figure)

        run_test_figure = plot_test_acc(run_df, config["epoch"])

        writer.add_figure("Run Test Accuracy", run_test_figure)

        run_test_figure = plot_mean_run_loss(mean_df, std_df, config["epoch"])

        writer.add_figure("AVG Loss", run_test_figure)

        run_test_figure = plot_mean_run_acc(mean_df, std_df, config["epoch"])

        writer.add_figure("AVG Test Accuracy", run_test_figure)

        run_quant_train_figures = plot_quant_train_acc(mean_df, std_df)

        for metric in run_quant_train_figures:
            writer.add_figure(f"Quantization Train {metric}", run_quant_train_figures[metric])

        run_quant_figures = plot_quant_acc(mean_df, std_df)

        for metric in run_quant_figures:
            writer.add_figure(f"Quantization {metric}", run_quant_figures[metric])

        for trial_id in self._curr_objective_metadata:
            for test_epoch in self._curr_objective_metadata[trial_id]:
                epoch_data = self._curr_objective_metadata[trial_id][test_epoch]
                if "quant_centers" in epoch_data:
                    for quant_level in epoch_data["quant_centers"]:
                        for layer in epoch_data["weights"]:
                            fig = plot_weights(epoch_data["weights"][layer], epoch_data["quant_centers"][quant_level])
                            writer.add_figure(f"epoch_{test_epoch}/{layer}/quant_level_{quant_level}", fig)
                else:
                    for layer in epoch_data["weights"]:
                        fig = plot_weights(epoch_data["weights"][layer])
                        writer.add_figure(f"epoch_{test_epoch}/{layer}", fig)

        writer.flush()
        writer.close()



    def opt_objective(self, trial = None):
        """
        Evaluate optimization objective
        """
        total_runs = 1 + self.study_config.reruns

        trial_id = str(trial.number) if trial else None


        run_data = []
        for run_num in range(total_runs):
            torch.cuda.empty_cache()
            runner_config = self.study_config.make_runner_config_dict(run_num, trial, trial_id)
            runner = create_runner(runner_config)
            runner.train()
            runner.save_runs_config()
            run_df = runner.run_stats_to_df()
            run_data.append(run_df)

        run_data = pd.concat(run_data)

        runner.save_run_data(run_data)

        print(run_data)

        mean_df, std_df, selected_vals = preprocess_run_data(run_data)

        print(selected_vals)
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
