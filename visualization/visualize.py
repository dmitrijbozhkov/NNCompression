#!/usr/bin/env python3
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import json

def combine_result_dataset(data_folder):
    data_path = Path(data_folder)

    run_datasets = []
    for trial_dir in data_path.iterdir():
        if trial_dir.is_dir():
            config_path = trial_dir / "config.json"
            run_data_path = trial_dir / "run_data.parquet"
            if not (config_path.is_file() and run_data_path.is_file()):
                continue
            trial_id = trial_dir.parts[-1]

            with open(config_path, "r") as f:
                config = json.load(f)

            search_config = {c: config["config"][c] for c in config["config"] if c in config["search_space"]}

            run_data = pd.read_parquet(run_data_path)

            for hp in search_config:
                run_data[hp] = search_config[hp]

            run_data["trial_idx"] = trial_id

            run_datasets.append(run_data)

    return pd.concat(run_datasets)


def plot_run_loss(run_df, epoch_amount):
    fig1 = plt.figure()

    for run_num in run_df["run_num"].unique():
        select_df = run_df[run_df["run_num"] == run_num]
        plt.plot(range(1, epoch_amount + 1), select_df["train_loss"], label=f'Training Loss Run {run_num}')
        plt.plot(range(1, epoch_amount + 1), select_df["valid_loss"], label=f'Validation Loss  Run {run_num}')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.xlim(0, epoch_amount + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    return fig1


def plot_test_acc(run_df, epoch_amount):
    fig1 = plt.figure()
    for run_num in run_df["run_num"].unique():
        select_df = run_df[run_df["run_num"] == run_num]
        plt.plot(range(1, epoch_amount + 1), select_df["accuracy"], label=f'Test Accuracy Run {run_num}')
        plt.xlabel('epochs')
        plt.ylabel('test accuracy')
        plt.xlim(0, epoch_amount + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    return fig1


def plot_quant_train_acc(mean_df, std_df):
    mean_df = mean_df[mean_df["epoch"] == 1]
    std_df = std_df.loc[mean_df.index]
    quant_data = [re.search(r"quant_(\d+)_(\w+)", c) for c in mean_df.columns if "quant" in c]
    quant_data = [(m.group(1), m.group(2)) for m in quant_data]
    metrics = set(q[1] for q in quant_data)

    metric_figs = {}
    for metric in metrics:
        fig = plt.figure()
        metric_quantizations = sorted([int(q_d[0]) for q_d in quant_data if q_d[1] == metric], reverse=False)
        for quant in metric_quantizations:
            quant_column = f"quant_{quant}_{metric}"
            # plt.errorbar(mean_df.index, mean_df[quant_column], std_df[quant_column], label=f'{quant}')
            plt.plot(mean_df.index, mean_df[quant_column], "-o")
            plt.fill_between(mean_df.index, mean_df[quant_column] - std_df[quant_column], mean_df[quant_column] + std_df[quant_column], alpha=0.15, label=f'{quant}')
            plt.xlabel('test epoch')
            plt.ylabel(f'quantization {metric}')
            plt.xticks(mean_df.index)
            plt.xlim(0, mean_df.index[-1] + 1)  # consistent scale
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

        metric_figs[metric] = fig

    return metric_figs


def plot_quant_acc(mean_df, std_df):
    quant_data = [re.search(r"quant_(\d+)_(\w+)", c) for c in mean_df.columns if "quant" in c]
    quant_data = [(m.group(1), m.group(2)) for m in quant_data]
    metrics = set(q[1] for q in quant_data)

    metric_figs = {}
    for metric in metrics:
        fig = plt.figure()
        metric_quantizations = sorted([int(q_d[0]) for q_d in quant_data if q_d[1] == metric], reverse=False)
        quant_last_mean = []
        quant_last_std = []
        for quant in metric_quantizations:
            quant_column = f"quant_{quant}_{metric}"
            quant_last_mean.append(mean_df[quant_column].iloc[-1])
            quant_last_std.append(std_df[quant_column].iloc[-1])

        quant_last_mean = np.array(quant_last_mean)
        quant_last_std = np.array(quant_last_std)
        quant_steps = [i for i in range(1, len(metric_quantizations) + 1)]
        plt.plot(quant_steps, quant_last_mean, "-o")
        plt.fill_between(quant_steps, quant_last_mean - quant_last_std, quant_last_mean + quant_last_std, alpha=0.15)
        plt.xlabel('quantization levels')
        plt.ylabel(f'quantization {metric}')
        plt.xticks(quant_steps, metric_quantizations)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        metric_figs[metric] = fig

    return metric_figs

def plot_mean_run_loss(mean_df, std_df, epoch_amount):
    fig1 = plt.figure()
    plt.errorbar(range(1, epoch_amount + 1), mean_df["train_loss"], std_df["train_loss"], capsize=3, label='Average train loss')
    plt.errorbar(range(1, epoch_amount + 1), mean_df["valid_loss"], std_df["valid_loss"], capsize=3, label='Average train loss')
    plt.xlabel('epochs')
    plt.ylabel('mean loss')
    plt.xlim(0, epoch_amount + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig1


def plot_mean_run_acc(mean_df, std_df, epoch_amount):
    fig1 = plt.figure()
    plt.errorbar(range(1, epoch_amount + 1), mean_df["accuracy"], std_df["accuracy"], capsize=3, label='Average test accuracy')
    plt.xlabel('epochs')
    plt.ylabel('mean acc')
    plt.xlim(0, epoch_amount + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig1


def plot_weights(weights, quant_clusters=None, bins=50):
    fig1 = plt.figure()
    plt.hist(weights, bins=bins)
    if quant_clusters is not None:
        plt.ylim(ymin=0)
        plt.scatter(quant_clusters, [0 for _ in range(len(quant_clusters))], color="red", clip_on=False, label="KMeans centers")
        plt.legend()
    plt.xlabel('weight')
    plt.ylabel('weight count')
    plt.tight_layout()

    return fig1


def plot_hesse(hesse, epoch, result_path):
    ax = sns.distplot((hesse.flatten()).cpu().detach().numpy(), color='b', kde=False, bins=20, hist_kws={'log': True})
    ax.set_xlabel('Values of second derivatives')
    ax.set_ylabel('frequences')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(10, 400000)
    dist_fig = ax.get_figure()
    dist_fig.savefig(result_path + '/hesse_distribution_{}.png'.format(epoch))
    plt.close()
    bx = sns.heatmap(hesse.cpu().detach().numpy(), vmin=-0.5, vmax=0.5)
    bx.set_xlabel('row')
    bx.set_ylabel('column')
    plt.title('Hesse Matrix')
    ax.xaxis.tick_top()
    heatmap_fig = bx.get_figure()
    heatmap_fig.savefig(result_path + '/hesse_{}.png'.format(epoch))
    plt.close()

def plot_quant_mean_perturb_acc(mean_orig, mean_perturb):
    quant_data = [re.search(r"quant_(\d+)_(\w+)", c) for c in mean_orig.columns if "quant" in c]
    quant_data = [(m.group(1), m.group(2)) for m in quant_data]
    metrics = set(q[1] for q in quant_data)

    metric_figs = {}
    for metric in metrics:
        fig = plt.figure()
        metric_quantizations = sorted([int(q_d[0]) for q_d in quant_data if q_d[1] == metric], reverse=False)
        quant_last_orig = []
        quant_last_perturb = []
        for quant in metric_quantizations:
            quant_column = f"quant_{quant}_{metric}"
            quant_last_orig.append(mean_orig[quant_column].iloc[-1])
            quant_last_perturb.append(mean_perturb[quant_column].iloc[-1])

        quant_last_orig = np.array(quant_last_orig)
        quant_last_perturb = np.array(quant_last_perturb)
        metric_quantizations = np.array(metric_quantizations)
        quant_steps = [i for i in range(1, len(metric_quantizations) + 1)]
        plt.plot(quant_steps, quant_last_orig, "-o", label="No Perturb")
        plt.plot(quant_steps, quant_last_perturb, "-o", label="Perturb")
        for j in range(len(quant_last_orig)):
            plt.text(quant_steps[j] - 0.6, quant_last_orig[j] + 3, np.round(quant_last_orig[j], 2))
            plt.text(quant_steps[j] + 0.1, quant_last_perturb[j] - 3, np.round(quant_last_perturb[j] - quant_last_orig[j], 2))
        # plt.fill_between(metric_quantizations, quant_last_orig - quant_last_perturb, quant_last_ + quant_last_std, alpha=0.15)
        plt.xlabel('quantization levels')
        plt.ylabel(f'quantization {metric}')
        plt.xticks(quant_steps, metric_quantizations)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        metric_figs[metric] = fig

    return metric_figs
