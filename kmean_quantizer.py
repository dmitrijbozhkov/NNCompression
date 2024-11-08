from kmeans_pytorch import kmeans, kmeans_predict
import operator
import torch
import torch.nn as nn


def kmeans_quantizer(net, level, device):
    params = net.parameters()
    for i, p in enumerate(params):
        if i == 0:
            weights = p.data.view(-1, 1)
        else:
            weights = torch.cat((weights, p.data.view(-1, 1)))

    cluster_ids_x, cluster_centers = kmeans(X=weights,
        num_clusters=level,
        device=device,
        tol=0.0001)

    param_ = net.named_parameters()
    for name, param in param_:
        netweight = operator.attrgetter(name)(net)
        netweight_label = kmeans_predict(netweight.reshape(-1, 1), cluster_centers, device=device)
        netweight_quanti = cluster_centers[netweight_label].reshape(netweight.size())
        net_change = operator.attrgetter(name)(net)
        net_change.data.copy_(nn.parameter.Parameter(netweight_quanti.type(torch.cuda.FloatTensor)))
