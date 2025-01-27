import torch.optim as optim

from models.spsa import SPSA


def get_optimizer(model, objective, device, runner_config: dict):
    """
    Initialize optimizer for training the neural network

    :param model: Neural network model to optimize
    :param objective: Objective loss module
    :param device: Device to use for optimization
    :param runner_config: Configuration object for Runner
    :returns: torch Optimizer
    """
    if runner_config["optimizer_type"] == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=runner_config["learning_rate"],
            momentum=runner_config["optimizer_momentum"],
            weight_decay=runner_config["weight_decay"],
            foreach=True
        )
    if runner_config["optimizer_type"] == "adam":
        return optim.Adam(
            model.parameters(),
            lr=runner_config["learning_rate"],
            betas=(runner_config["optimizer_adam_beta_1"], runner_config["optimizer_adam_beta_2"]),
            eps=runner_config["optimizer_eps"],
            weight_decay=runner_config["weight_decay"],
            foreach=True
        )
    if runner_config["optimizer_type"] == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=runner_config["learning_rate"],
            betas=(runner_config["optimizer_adam_beta_1"], runner_config["optimizer_adam_beta_2"]),
            eps=runner_config["optimizer_eps"],
            weight_decay=runner_config["weight_decay"],
            foreach=True
        )
    if runner_config["optimizer_type"] == "spsa":
        lr_stabilizer = runner_config["optimizer_spsa_lr_stabilizer"]
        if lr_stabilizer is None:
            lr_stabilizer = runner_config["epoch"] * 0.1
        return SPSA(
            model,
            objective,
            device,
            initial_lr=runner_config["optimizer_spsa_initial_lr"],
            initial_perturb_magnitude=runner_config["optimizer_spsa_initial_perturb_magnitude"],
            lr_stabilizer=lr_stabilizer,
            lr_decay=runner_config["optimizer_spsa_lr_decay"],
            perturb_decay=runner_config["optimizer_spsa_perturb_decay"]
        )

def get_scheduler(optimizer, runner_config: dict):
    """
    Initialize scheduler for training the neural network

    :param optimizer: model optimizer
    :param runner_config: Configuration object for runner
    :returns: torch LRScheduler
    """
    if runner_config["scheduler_type"] == "const":
        return None
    if runner_config["scheduler_type"] == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=runner_config["scheduler_milestones"],
            last_epoch=-1
        )
    if runner_config["scheduler_type"] == "lambda":
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda e: runner_config["scheduler_lambda_multiplier"] if e < runner_config["scheduler_last_epoch"] else 1,
        )
    if runner_config["scheduler_type"] == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            runner_config["scheduler_step_size"],
            runner_config["scheduler_gamma"],
        )
    if runner_config["scheduler_type"] == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            runner_config["scheduler_T_max"],
            runner_config["scheduler_eta_min"],
        )
