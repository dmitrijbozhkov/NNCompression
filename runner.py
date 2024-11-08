#!/usr/bin/env python3
from operator import attrgetter
from utils import set_device, get_data, get_model, get_scheduler, get_optimizer
from objectives import HessianCELoss
from kmean_quantizer import kmeans_quantizer
from copy import deepcopy
import hashlib
import os
import torch.optim as optim
import time
import torch
import numpy as np
import pandas as pd
import json

from visualize import plot_loss, plot_weights

class Runner:
    """Runner class for network"""

    def __init__(self,
                 net,
                 optimizer,
                 scheduler,
                 dataset,
                 objective,
                 config,
                 result_path,
                 device) -> None:
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.objective = objective
        self.config = config
        self.result_path = result_path
        self.checkpoint_path = self.result_path / "checkpoints"
        self.device = device

    @classmethod
    def from_config(cls, config: dict, trial_id = None):
        """
        Create runenr from config

        :param config: Configuration dict
        """
        device = set_device(config["gpu"])
        dataset = get_data(config["dataset"], config["batch_size"])
        model = get_model(config["model"], device)
        optimizer = get_optimizer(model.parameters(), config)
        scheduler = get_scheduler(optimizer, config)
        # scheduler = get_
        objective = HessianCELoss(model, config)

        if trial_id is not None:
            run_folder = str(trial_id)
        else:
            conf_path = deepcopy(config)
            conf_path["output"] = conf_path["output"].as_posix()
            conf_string = json.dumps(conf_path)
            run_folder = hashlib.md5(conf_string.encode()).hexdigest()

        result_path = config["output"] / config["study_name"] / run_folder
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            os.makedirs(result_path / "checkpoints")

        return cls(model, optimizer, scheduler, dataset, objective, config, result_path, device)


    def save_checkpoint(self, checkpoint_name):
        """
        Save model checkpoint on disk
        """
        torch.save(self.net.state_dict(), self.checkpoint_path / f"{checkpoint_name}.pth")


    def load_checkpoint(self, checkpoint_name: str):
        """
        Load model checkpoint
        """
        self.net.load_state_dict(torch.load(self.checkpoint_path / f"{checkpoint_name}.pth"))


    def save_train_run(self, run_dataframe: pd.DataFrame):
        """
        Saves train run data
        """
        run_dataframe.to_csv(self.result_path / "train_timeline.csv")
        with open(self.result_path / "config.json", "w") as f:
            config = self.config
            config["output"] = str(config["output"])
            json.dump(config, f, indent=4)


    def save_plots(self, run_df: pd.DataFrame):
        """
        Create plots and save them
        """
        plot_loss(
            run_df["train_loss"],
            run_df["valid_loss"],
            run_df["train_cross_entropy"],
            run_df["valid_cross_entropy"],
            result_path=self.result_path)

        plot_weights(
            self.net,
            self.result_path
        )

    def perturb_batch(self, batch):
        """
        Create perturbations for network
        """
        outputs = []

        for i in range(self.config["perturb_amount"]):
            perturb_params = {}
            for param in self.config["perturb_param"]:
                net_params = attrgetter(param)(self.net)
                perturb_params[param] = torch.normal(
                    self.config["perturb_mean"],
                    torch.full_like(net_params, self.config["perturb_variance"])
                )

            output = self.net.forward(batch, perturb_params)
            outputs.append(output)

        return torch.cat(outputs, 0)


    def train(self, epochs=None, save_checkpoints=True):
        """
        Train model
        """
        train_losses, valid_losses, train_cross_entropies, valid_cross_entropies = [], [], [], []
        avg_train_losses, avg_valid_losses, avg_train_cross_entropies, avg_valid_cross_entropies = [], [], [], []
        epoch_elapsed, test_accuracy = [], []
        if not epochs:
            epochs = self.config["epoch"]
        # initialize the early_stopping object
        '''early_stopping = EarlyStopping(patience=5, verbose=True, path=result_path + '/checkpoint.pt', gpu=CFG.gpu)'''
        for epoch in range(1, epochs + 1):
            epoch_elapsed.append(epoch)
            self.net.train()
            for batch_idx, (data, target) in enumerate(self.dataset.train_loader):
                s = time.time()
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                if self.config["is_perturb"] and epoch >= self.config["perturb_start"]:
                    perturb = self.perturb_batch(data)
                else:
                    perturb = None
                output = self.net.forward(data).to(self.device)
                loss, cross_entropy, curvature = self.objective(output, target, perturb)
                loss.backward()
                # print(next(net.get_model_gradients()))
                self.optimizer.step()
                train_losses.append(loss.item())
                train_cross_entropies.append(cross_entropy)
                e = time.time()
                """
                if batch_idx == 0 and epoch == 1:
                    print("1 batch time:{:.0f}s".format(e - s))
                if batch_idx % 20 == 0:
                    print('Train Batch: [{}/{}]  train_loss: {:.6f} train_CrossEntropy: {:.6f} '
                    'Curvature: {:.6f}'.format(batch_idx, len(self.dataset.train_loader), loss.item(), cross_entropy, curvature))
                """
            self.net.eval()  # prep model for evaluation
            torch.cuda.empty_cache()
            for batch_idx, (data, target) in enumerate(self.dataset.valid_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.net(data).to(self.device)
                loss, cross_entropy, curvature = self.objective(output, target)
                # record validation loss
                valid_losses.append(loss.item())
                valid_cross_entropies.append(cross_entropy)
            if self.scheduler:
                self.scheduler.step()
            train_loss, valid_loss = np.average(train_losses), np.average(valid_losses)
            train_cross_entropy, valid_cross_entropy = np.average(train_cross_entropies), np.average(valid_cross_entropies)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            avg_train_cross_entropies.append(train_cross_entropy)
            avg_valid_cross_entropies.append(valid_cross_entropy)
            """
            print('Train Epoch: [{}/{}]  train_loss: {:.6f} valid_loss: {:.6f} train_CrossEntropy: {:.6f} '
                'valid_CrossEntropy: {:.6f} Curvature: {:.6f}'.format(epoch, epochs, train_loss, valid_loss,
                                                                        train_cross_entropy, valid_cross_entropy,
                                                                        curvature))
            """
            # clear lists to track next epoch
            train_losses, valid_losses, train_cross_entropies, valid_cross_entropies = [], [], [], []
            test_loss, crossentropy, accuracy = self.test()
            test_accuracy.append(accuracy)
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            '''early_stopping(valid_cross_entropy, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break'''
            '''e = time.time()
            print("1 epoch time:{:.0f}s".format(e - s))'''

        # load the last checkpoint with the best model
        '''if CFG.gpu:
            net.module.load_state_dict(torch.load(result_path + '/checkpoint.pt'))
        else:
            net.load_state_dict(torch.load(result_path + '/checkpoint.pt'))'''
        train_df = pd.DataFrame({
            "epoch": epoch_elapsed,
            "train_loss": avg_train_losses,
            "valid_loss": avg_valid_losses,
            "train_cross_entropy": avg_train_cross_entropies,
            "valid_cross_entropy": avg_valid_cross_entropies,
            "accuracy": test_accuracy
        })
        return train_df


    def test(self):
        self.net.eval()
        test_loss = []
        curvature = 0
        crossentropy = 0
        correct = 0
        for data, target in self.dataset.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.net(data).to(self.device)
            loss, cr, cu = self.objective(output, target)
            test_loss.append(loss.item())
            crossentropy += cr
            curvature += cu
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss = np.average(test_loss)
        accuracy = 100. * correct / len(self.dataset.test_loader.dataset)
        crossentropy = crossentropy / len(self.dataset.test_loader)
        curvature = curvature / len(self.dataset.test_loader)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                len(self.dataset.test_loader), accuracy))
        return test_loss, crossentropy, accuracy


    def kmeans_quantize(self, level):
        """
        Quantize network using KMeans
        """
        kmeans_quantizer(self.net, level, self.device)
