from copy import deepcopy
import pandas as pd
import json
import hashlib
import torch

def get_device(runner_config: dict):
    if not runner_config["gpu"]:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(runner_config["gpu"])


def get_last_run_epochs(run_df):
    """
    Get last epochs for earch run
    :param run_df: Pandas dataframe with run data
    :returns: DataFrame with rows of last epoch for each run
    """
    run_max_epochs = run_df.groupby("run_num")["epoch_num"].max().reset_index()
    return pd.merge(run_df, run_max_epochs, on=["run_num", "epoch_num"])

def get_run_stats(run_df, group_by_epoch=True):
    """
    Get mean and standard deviation of run stat dataframe

    :param run_df: Pandas dataframe with run data
    :param group_by_epoch: Should rows be grouped by epoch_num column
    :returns: Tuple of mean and std DataFrame if group_by_epoch is true or Series if False
    """
    if group_by_epoch:
        mean_df = run_df.groupby("epoch_num").mean()
        std_df = run_df.groupby("epoch_num").std()
    else:
        mean_df = run_df.mean()
        std_df = run_df.std()
    return (mean_df, std_df)


def get_mean_std_run(run_df):
    epoch_amount = run_df["test_epoch"].map(
        run_df[run_df["run_num"] == 0]["test_epoch"].value_counts()
    )
    run_df["total_idx"] = run_df["test_epoch"] + run_df["epoch"] - epoch_amount
    mean_df = run_df.groupby(["total_idx"]).mean()
    std_df = run_df.groupby(["total_idx"]).std()

    return mean_df, std_df


class TrialConfig(dict):
    """
    Configuration object for trial
    """

    def __init__(self, config, search_space=None, trial=None):
        if not search_space:
            search_space = {}

        cc = {c: config[c] for c in config if c not in search_space}
        super().__init__(**cc)
        self.search_space = search_space
        self.trial = trial

    def to_serializable(self):
        conf_path = {k: self[k] for k in self}
        conf_path["output"] = conf_path["output"].as_posix()
        return conf_path

    def to_valid_hp(self):
        conf = self.to_serializable()
        del conf["quantize_levels"]
        del conf["perturb_param"]
        return conf

    def to_full_config(self):
        conf = self.to_serializable()
        search_space = deepcopy(self.search_space)
        return {
            "config": conf,
            "search_space": search_space
        }

    def config_to_id(self):
        conf_path = self.to_serializable()
        conf_string = json.dumps(conf_path)
        return hashlib.md5(conf_string.encode()).hexdigest()

    def __getitem__(self, key):

        if key in self:
            return super().__getitem__(key)

        if not (self.search_space and self.trial):
            raise KeyError("Search space and trial not set!")

        if isinstance(self.search_space[key], list):
            sampled_value = self.trial.suggest_categorical(
                key,
                self.search_space[key]
            )
        elif self.search_space[key]["type"] == "categorical":
            sampled_value = self.trial.suggest_categorical(
                key,
                self.search_space[key]["choices"]
            )
        elif self.search_space[key]["type"] == "int":
            sampled_value = self.trial.suggest_int(
                key,
                self.search_space[key]["low"],
                self.search_space[key]["high"],
                step=self.search_space[key].get("step", 1),
                log=self.search_space[key].get("log", False)
            )
        elif self.search_space[key]["type"] == "float":
            sampled_value = self.trial.suggest_float(
                key,
                self.search_space[key]["low"],
                self.search_space[key]["high"],
                step=self.search_space[key].get("step", None),
                log=self.search_space[key].get("log", False)
            )
        else:
            raise KeyError(f"Wrong hyperparameter type: {self.search_space[key]['type']}")

        self[key] = sampled_value
        return sampled_value
