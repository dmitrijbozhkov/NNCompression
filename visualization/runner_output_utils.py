"""Utilities for manipulating runner statistics"""

from os.path import isdir, isfile
from pathlib import Path
import json
import os
import shutil
import pandas as pd


def extract_model_weights(model, weight_name="weight"):
    modules = [module for module in model.named_modules()]
    modules = modules[1:]
    module_weights = {}
    getter = attrgetter(weight_name)
    for name, layer in modules:
        if hasattr(layer, weight_name):
            w = getter(layer).data.cpu().numpy().flatten()
            module_weights[name] = w
    return module_weights


def extract_model_grads(model, weight_name="weight"):
    modules = [module for module in model.named_modules()]
    modules = modules[1:]
    module_weights = {}
    getter = attrgetter(weight_name)
    for name, layer in modules:
        if hasattr(layer, weight_name):
            w = getter(layer).data.cpu().numpy().flatten()
            module_weights[name] = w
    return module_weights


def get_mean_std_stat(run_df):
    """
    Average each rerun and return the value
    """
    functional_columns = ["epoch_num", "run_num", "time", "train_loss"]
    data_columns = [c for c in run_df.columns if c not in functional_columns]
    data_columns = [c for c in data_columns if "centers" not in c]

    mean_df = run_df.groupby("epoch_num")[data_columns].mean().dropna().reset_index()
    std_df = run_df.groupby("epoch_num")[data_columns].std().dropna().reset_index()

    return mean_df, std_df

def preprocess_run_data(run_df):
    """
    Preprocess run data for visualization or manipulation
    """
    mean_df, std_df = get_mean_std_stat(run_df)


    selected_vals = {}
    for c in mean_df.columns:
        if c == "epoch_num":
            continue
        selected_vals[f"{c}_max"] = mean_df[c].max()
        selected_vals[f"{c}_last"] = mean_df[c].iloc[-1].item()

    return mean_df, std_df, selected_vals


def enumerate_run_folder(folder_path):
    """
    Populate dictionary with model runs

    :param folder_path: Path to experiment run folder
    :returns: Dictionary of runs and paths
    """
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)

    run_folders = {}
    for folder in folder_path.iterdir():
        if os.path.isdir(folder):
            curr_folder = {}
            run_data_path = folder / "run_data.parquet"
            if os.path.isfile(run_data_path):
                curr_folder["run_data"] = run_data_path
            config_path = folder / "config.json"
            if os.path.isfile(config_path):
                curr_folder["config"] = config_path
            checkpoint_path = folder / "checkpoints"
            if os.path.isdir(checkpoint_path):
                curr_folder["checkpoint_path"] = checkpoint_path
            run_folders[folder.name] = curr_folder

    return run_folders


def filter_incomplete_runs(run_dict):
    """
    Filter out incomplete runs that don't have run_data or config

    :param run_dict: Dictionary with run data
    :returns: Run dictionary without empty runs
    """
    run_complete = {}

    for run in run_dict:
        if run_dict[run].get("run_data") and run_dict[run].get("config"):
            run_complete[run] = run_dict[run]

    return run_complete


def filter_run_folder(folder_dict, config_key, config_value):
    """
    Filter run dictionary by config key and value

    :param fodler_dict: Dictionary of runs
    :param config_key: Run configuration key
    :param config_value: Run configuration value
    :returns: Dictionary of runs that is filtered
    """

    filtered_dicts = {}
    for run in folder_dict:
        run_data = folder_dict[run]
        if "config" in run_data:
            with open(run_data["config"], "r") as f:
                config = json.load(f).get("config")

                if not config:
                    continue

                if config_key not in config:
                    continue

                if config[config_key] == config_value:
                    filtered_dicts[run] = run_data

    return filtered_dicts


def get_config_run_data(run_dict):
    """
    Reads and returns config and run_data

    :param run_dict: Dictionary of run parameters
    """

    run_data = pd.read_parquet(run_dict["run_data"])

    with open(run_dict["config"], "r") as f:
        config = json.load(f)["config"]

    return config, run_data



def copy_results(folder_dict, target_folder):
    """
    Copy only results into the folder without checkpoints

    :param folder_dict: Dictionary with data folders
    :param target_folder: target folder for data
    """
    if not isinstance(target_folder, Path):
        target_folder = Path(target_folder)

    for run_id in folder_dict:
        target_path = target_folder / run_id
        os.mkdir(target_path)

        config = folder_dict[run_id].get("config")
        if config:
            shutil.copyfile(config, target_path / "config.json")

        run_data = folder_dict[run_id].get("run_data")
        if run_data:
            shutil.copyfile(run_data, target_path / "run_data.parquet")
