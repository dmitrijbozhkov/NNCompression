"""Utilities for manipulating runner statistics"""

from pathlib import Path


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


def enumerate_run_fodler(folder_path):
    if not isinstance(folder_path, Path):
        fodler_path = Path(folder_path)
