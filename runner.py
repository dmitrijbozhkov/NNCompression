#!/usr/bin/env python3
from operator import attrgetter

from torch.utils.tensorboard.writer import SummaryWriter
from utils import set_device, get_data, get_model, get_scheduler, get_optimizer
from objectives import HessianCELoss
from quantization import kmeans_quantizer
import os
import time
import torch
import numpy as np
import pandas as pd
import json

class Runner:
    """Runner class for network"""

    def __init__(self,
                 net,
                 optimizer,
                 scheduler,
                 dataset,
                 objective,
                 summary_writer,
                 config,
                 result_path,
                 run_num,
                 device) -> None:
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.objective = objective
        self.summary_writer = summary_writer
        self.config = config
        self.result_path = result_path
        self.checkpoint_path = self.result_path / "checkpoints" / str(run_num)
        self.total_epoch = 0
        self.device = device

    @classmethod
    def from_config(cls, config: dict, study_writer: SummaryWriter, trial_id, run_num):
        """
        Create runenr from config

        :param config: Configuration dict
        """
        device = set_device(config["gpu"])
        dataset = get_data(config["dataset"], config["batch_size"])
        model = get_model(config["model"], device)
        optimizer = get_optimizer(model.parameters(), config)
        scheduler = get_scheduler(optimizer, config)
        objective = HessianCELoss(model, config)

        result_path = cls.runs_path(config, trial_id)
        os.makedirs(result_path / "checkpoints" / str(run_num))

        # study_writer.add_graph(model)

        return cls(model, optimizer, scheduler, dataset, objective, study_writer, config, result_path, run_num, device)



    def perturb_batch(self, batch):
        """
        Create perturbations for network
        """
        outputs = []

        for i in range(self.config["perturb_amount"]):
            perturb_params = {}
            for param in self.config["perturb_param"]:
                net_params = attrgetter(param)(self.net)
                # perturb_params[param] = self.config["perturb_lr"] *
                perturb_params[param] = torch.normal(
                    self.config["perturb_mean"],
                    torch.full_like(net_params, self.config["perturb_variance"])
                )

            output = self.net.forward(batch, perturb_params)
            outputs.append(output)

        return outputs


    def train(self, epochs=None, metadata=None):
        """
        Train model for given number of epochs

        :param epochs: Number of epochs to train the model
        :param metadata: List of metadata to return from training
        """
        if metadata is None:
            metadata = []
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
                if self.config["is_perturb"] and (self.total_epoch + epoch) >= self.config["perturb_start"]:
                    perturb = self.perturb_batch(data)
                else:
                    perturb = None
                output = self.net.forward(data).to(self.device)
                loss, cross_entropy, curvature = self.objective(output, target, perturb)
                loss.backward()
                # print(next(net.get_model_gradients()))
                # print([(n, p.grad) for n, p in self.net.named_parameters()])
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
            accuracy = self.test_accuracy()
            test_accuracy.append(accuracy)

        self.total_epoch += epochs
        train_df = pd.DataFrame({
            "epoch": epoch_elapsed,
            "train_loss": avg_train_losses,
            "valid_loss": avg_valid_losses,
            "train_cross_entropy": avg_train_cross_entropies,
            "valid_cross_entropy": avg_valid_cross_entropies,
            "accuracy": test_accuracy
        })
        return train_df


    def test_accuracy(self):
        self.net.eval()
        correct = 0
        for data, target in self.dataset.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.net(data).to(self.device)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(self.dataset.test_loader.dataset)

        return accuracy


    def kmeans_quantize(self, level):
        """
        Quantize network using KMeans
        """
        return kmeans_quantizer(self.net, level, self.device)
