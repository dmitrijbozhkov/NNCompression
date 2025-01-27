#!/usr/bin/env python3

import torch
import torch.nn as nn


class SPSA:
    """SPSA optimizer for neural networks"""


    def __init__(self,
                 model,
                 objective: nn.Module,
                 device,
                 initial_lr: float,
                 initial_perturb_magnitude: float,
                 lr_stabilizer: int,
                 lr_decay=0.602,
                 perturb_decay=0.101) -> None:
        self.model = model
        self.objective = objective
        self.device = device
        # SPSA hyperparameters
        self.alpha = lr_decay # Learning rate decay
        self.gamma = perturb_decay # Perturbation decay
        self.a = initial_lr # initial change value
        self.A = lr_stabilizer # N_iterations * 0.1
        self.c = initial_perturb_magnitude

        #SPSA parameters
        self.curr_step = 1
        self.set_spsa_vector()


    @torch.no_grad()
    def set_spsa_vector(self):
        vector_len = 0
        for param in self.model.parameters():
            vector_len += param.flatten().shape[0]

        self.spsa_params = torch.zeros((vector_len,), device=self.device)

        pointer = 0
        for param in self.model.parameters():
            param_vecotr = param.flatten()
            pointer_end = pointer + param_vecotr.shape[0]
            self.spsa_params[pointer:pointer_end] = param_vecotr
            pointer = pointer_end


    @torch.no_grad
    def set_model_from_vector(self, vector):
        pointer = 0
        for param in self.model.parameters():
            pointer_end = pointer + param.flatten().shape[0]
            param.copy_(vector[pointer:pointer_end].reshape(param.shape))
            pointer = pointer_end


    @torch.no_grad
    def model_eval(self, data_loader):
        """
        Evaluate model with given data
        """
        self.model.eval()
        batch_objectives = []
        for data, target in data_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model.forward(data)
            objective = self.objective(output, target)
            batch_objectives.append(objective)

        batch_objectives = torch.cat(batch_objectives)

        return torch.mean(batch_objectives)


    def step(self, data_loader):
        a_n = self.a / (self.A + self.curr_step + 1) ** self.alpha
        c_n = self.c / (self.curr_step + 1) ** self.gamma

        draws = torch.empty(self.spsa_params.shape, device=self.device).uniform_(0, 1)
        draws = torch.bernoulli(draws)
        draws[draws == 0] = -1

        weights_perturb = self.spsa_params + c_n * draws

        self.set_model_from_vector(weights_perturb)

        forward_difference = self.model_eval(data_loader)

        weights_perturb = self.spsa_params - c_n * draws

        self.set_model_from_vector(weights_perturb)

        backward_difference = self.model_eval(data_loader)

        central_difference = (forward_difference - backward_difference) / (2 * c_n * draws)

        self.spsa_params = self.spsa_params - a_n * central_difference

        self.spsa_params.clamp_(-9999999, 9999999)

        self.curr_step += 1
