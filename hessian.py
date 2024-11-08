#!/usr/bin/env python3
import torch


def hessian(first_grad, net, device):
    cnt = 0
    for fg in first_grad:
        if cnt == 0:
            first_vector = fg.contiguous().view(-1)
            cnt = 1
        else:
            first_vector = torch.cat([first_vector, fg.contiguous().view(-1)])
    weights_number = first_vector.size(0)
    hessian_matrix = torch.zeros(weights_number, weights_number).to(device)
    for idx in range(weights_number):
        print(idx)
        second_grad = torch.autograd.grad(first_vector[idx], net.parameters(), create_graph=True, retain_graph=True)
        cnt = 0
        for g in second_grad:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian_matrix[idx] = g2
    return hessian_matrix


def hessian_topk(first_grad, net, k):
    cnt = 0
    for fg in first_grad:
        first_vector = fg.view(-1) if cnt == 0 else torch.cat([first_vector, fg.view(-1)])
        cnt = 1
    weights_number = first_vector.size(0)
    curvature = []
    for idx in range(weights_number):
        second_grad = torch.autograd.grad(first_vector[idx], net.parameters(), create_graph=True, retain_graph=True)
        cnt = 0
        for sg in second_grad:
            second_vector = sg.contiguous().view(-1) if cnt == 0 else torch.cat(
                [second_vector, sg.contiguous().view(-1)])
            cnt = 1
        second_topk = torch.topk(second_vector, k=int((weights_number - idx) * k + 1))[0]

        curvature.append(torch.pow(second_topk, 2))
        '''curvature = torch.sum(torch.pow(second_topk, 2)) if idx == 0 else curvature + torch.sum(
            torch.pow(second_topk, 2))'''

        # print('{}/{}'.format(idx,weights_number))
    return torch.sum(torch.tensor(curvature))


def hessian_random_topk(first_grad, net, k):
    cnt = 0
    for g in first_grad:
        first_vector = g.view(-1) if cnt == 0 else torch.cat([first_vector, g.view(-1)])
        cnt = 1
    weights_number = first_vector.size(0)
    random_idx = torch.randint(0, weights_number - 1, (int(0.001 * weights_number),))
    for idx in random_idx:
        second_grad = torch.autograd.grad(first_vector[idx], net.parameters(), create_graph=True, retain_graph=True)
        cnt = 0
        for g in second_grad:
            second_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([second_vector, g.contiguous().view(-1)])
            cnt = 1
        curvature = torch.sum(torch.pow(torch.topk(second_vector, k=int((weights_number - idx) * k + 1))[0], 2)) \
            if idx == random_idx[0] else curvature + torch.sum(
            torch.pow(torch.topk(second_vector[idx:], k=int((weights_number - idx) * k + 1))[0], 2))
        # print('{}/{}'.format(idx,weights_number))
    return curvature


def hessian_diagonal(first_grad, net,device):
    second_grad = []
    for i, parm in enumerate(net.parameters()):
        second_grad.append(torch.autograd.grad(first_grad[i], parm, retain_graph=True, create_graph=True,
                                               grad_outputs=torch.ones_like(first_grad[i]))[0])
    curvature = torch.tensor(0)
    for i in second_grad:
        curvature = curvature + torch.sum(torch.pow(i, 2))
    return curvature
