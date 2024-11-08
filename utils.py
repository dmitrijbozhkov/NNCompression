from collections import defaultdict, namedtuple
import torch
from torchvision import transforms, datasets
from models import *
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import math

dataset = namedtuple("dataset", ["train_loader", "test_loader", "valid_loader"])

def set_device(gpu=True):
    """SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)"""
    if gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

class TrialConfig(dict):
    """
    Configuration object for trial
    """

    def __init__(self, config, search_space=None, trial=None):
        if search_space and trial:
            fixed_config = {key: config[key] for key in config if key not in search_space}
        else:
            fixed_config = config
        super().__init__(**fixed_config)
        self.search_space = search_space
        self.trial = trial

    def __getitem__(self, key):

        if key in self:
            return super().__getitem__(key)

        if not (self.search_space and self.trial):
            raise KeyError("Search space and trial not set!")

        if self.search_space[key]["type"] == "categorical":
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


def get_data(name, batch_size):
    transform = transforms.ToTensor()
    if name == 'cifar10':
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    if name == 'MNIST':
        train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    if name == 'FashionMNIST':
        train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    # percentage of training set to use as validation
    valid_size = 0.2
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # load training data in batches
    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=12)
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=12)
    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=12)
    return dataset(train_loader, test_loader, valid_loader)


def get_model(name, device):
    if name == "LeNet3":
        net = LeNet3().to(device)
    if name == "LeNet3_2":
        net = LeNet3_2().to(device)
    if name == "LeNet3_3":
        net = LeNet3_3().to(device)
    if name == "LeNet3_4":
        net = LeNet3_4().to(device)
    if name == "LeNet_5":
        net = LeNet_5().to(device)
    if name == "alexnet":
        net = AlexNet().to(device)

    return net

def get_optimizer(parameters, config):
    """
    Initialize optimizer
    """
    if config["optimizer_type"] == "SGD":
        return optim.SGD(
            parameters,
            lr=config["learning_rate"],
            momentum=config["optimizer_momentum"],
            weight_decay=config["weight_decay"],
            foreach=True
        )
    if config["optimizer_type"] == "Adam":
        return optim.Adam(
            parameters,
            lr=config["learning_rate"],
            betas=(config["optimizer_adam_beta_1"], config["optimizer_adam_beta_2"]),
            eps=config["optimizer_eps"],
            weight_decay=config["weight_decay"],
            foreach=True
        )
    if config["optimizer_type"] == "AdamW":
        return optim.AdamW(
            parameters,
            lr=config["learning_rate"],
            betas=(config["optimizer_adam_beta_1"], config["optimizer_adam_beta_2"]),
            eps=config["optimizer_eps"],
            weight_decay=config["weight_decay"],
            foreach=True
        )

def get_scheduler(optimizer, config):
    """
    Initialize scheduler
    """
    if config["scheduler_type"] == "None":
        return None
    if config["scheduler_type"] == "Lambda":
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda e: config["scheduler_lambda_multiplier"] if e < config["scheduler_last_epoch"] else 1,
        )
    if config["scheduler_type"] == "Step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            config["scheduler_step_size"],
            config["scheduler_gamma"],
        )
    if config["scheduler_type"] == "Cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            config["scheduler_T_max"],
            config["scheduler_eta_min"],
        )


def quantiz_test(net, dataset, batch_size, device):
    net.eval()
    train_loader, test_loader, valid_loader = getData(dataset, batch_size)
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data).to(device)
        t = criterion(output, target)
        test_loss += t.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset),
                                                                              accuracy))
    return test_loss, accuracy


def compute_entropy(labels, base=None):
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = torch.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = torch.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = 2 if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent


def pdf(bundies, weights):
    cluster = torch.zeros_like(weights)
    prob = torch.zeros(len(bundies) - 1)
    for w in range(len(weights)):
        weight = weights[w]
        for b in range(len(bundies) - 1):
            if bundies[b] <= weight <= bundies[b + 1]:
                prob[b] += 1
                cluster[w] = b
    prob = prob / len(weights)
    return cluster, prob

