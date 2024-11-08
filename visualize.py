#!/usr/bin/env python3
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(train_loss, valid_loss, train_cross_entropy, valid_cross_entropy, result_path):
    fig1 = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')
    # find position of lowest validation loss
    '''minposs = valid_cross_entropy.index(min(valid_cross_entropy)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')'''
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig1.savefig(result_path / 'loss.png', bbox_inches='tight')
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_cross_entropy) + 1), train_cross_entropy, label='Training Cross Entropy')
    plt.plot(range(1, len(valid_cross_entropy) + 1), valid_cross_entropy, label='Validation Cross Entropy')
    # find position of lowest validation loss
    '''minposs = valid_cross_entropy.index(min(valid_cross_entropy)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')'''
    plt.xlabel('epochs')
    plt.ylabel('Cross Entropy')
    # plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_cross_entropy) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig2.savefig(result_path / 'CrossEntropy.png', bbox_inches='tight')
    plt.close(fig2)


def plot_weights(model, plot_path=None):
    modules = [module for module in model.modules()]
    num_sub_plot = 0
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            plt.subplot(131 + num_sub_plot)
            w = layer.weight.data
            w_one_dim = w.cpu().numpy().flatten()
            plt.hist(w_one_dim[w_one_dim != 0], bins=50)
            num_sub_plot += 1
    if plot_path:
        plt.savefig(plot_path)
    plt.show()


def plot_all_weights(model, result_path):
    modules = [module for module in model.modules()]
    modules = modules[1:]
    all_weights = torch.tensor([])
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            w = layer.weight.data.view(-1).cpu()
            all_weights = torch.cat((all_weights, w))
    sns.histplot(all_weights.cpu().numpy(),bins=50)
    plt.savefig(result_path / "all_weights.png", bbox_inches='tight')
    plt.close()


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


if __name__ == "__main__":
    lloyd_df_path = "./result/LeNet3_3_MNIST/time5/lambda_1.0/quantiz/lloydquantizer_30.csv"
    uniform_df_path = "./result/LeNet3_3_MNIST/time5/lambda_1.0/quantiz/uniformdquantizer_30.csv"
    lloyd_df = pd.read_csv(lloyd_df_path, index_col=0).reset_index()
    uniform_df = pd.read_csv(uniform_df_path, index_col=0).reset_index()
    print(uniform_df)
    fig = plt.figure(figsize=(10, 8))
    plt.title("Quantization pefrormance for LeNet3_3 on MNIST")
    plt.plot(lloyd_df["acc_after"], label="Lloyd quantizer")
    plt.plot(uniform_df["acc_after"], label="Uniform quantizer")
    plt.xticks(list(range(len(lloyd_df))), lloyd_df["index"])
    plt.xlabel('compression rate')
    plt.ylabel('accuracy on test set')
    plt.grid(True)
    # plt.tight_layout()
    plt.legend()
    fig.savefig("./compression.png", bbox_inches='tight')
