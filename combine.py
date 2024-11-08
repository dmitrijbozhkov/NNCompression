import torch.optim as optim
import argparse
from utils import *
import time
import os, sys
from uniform_quantizer import *

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--epoch", default=20, type=int)
parser.add_argument("--l", default=1.0, help="lambda", type=float)
parser.add_argument("--gpu", default=True, type=bool)
parser.add_argument("--dataset", default='MNIST')
parser.add_argument("--model", default='LeNet3')
parser.add_argument("--topk", default=1.0, type=float)  # if topk == 1.0 use the whole hesse matrix
parser.add_argument("--time", default='combine2', type=str)
CFG = parser.parse_args()

result_path = './result/{}_{}/time{}/lambda_{}'.format(CFG.model, CFG.dataset, CFG.time, CFG.l)
if not os.path.exists(result_path):
    os.makedirs(result_path)
    os.makedirs(result_path+'/pic')


def train(net, epochs):
    train_losses, valid_losses, train_cross_entropies, valid_cross_entropies = [], [], [], []
    avg_train_losses, avg_valid_losses, avg_train_cross_entropies, avg_valid_cross_entropies = [], [], [], []
    # initialize the early_stopping object
    for epoch in range(1, epochs + 1):
        net.train()
        '''s = time.time()'''
        if epoch > 10:
            CFG.l = 0.999
        for batch_idx, (data, target) in enumerate(train_loader):
            s = time.time()
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net.forward(data).to(device)
            loss, cross_entropy, curvature = my_loss_function(output, target, batch_idx, epoch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_cross_entropies.append(cross_entropy)
            e = time.time()
            if batch_idx == 0 and epoch == 1:
                print("1 batch time:{:.0f}s".format(e - s))
            '''if batch_idx % 20 == 0:
                print('Train Batch: [{}/{}]  train_loss: {:.6f} train_CrossEntropy: {:.6f} '
                  'Curvature: {:.6f}'.format(batch_idx, len(train_loader), loss.item(), cross_entropy, curvature))'''
        net.eval()  # prep model for evaluation
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            output = net(data).to(device)
            loss, cross_entropy, curvature = my_loss_function(output, target)
            # record validation loss
            valid_losses.append(loss.item())
            valid_cross_entropies.append(cross_entropy)
        train_loss, valid_loss = np.average(train_losses), np.average(valid_losses)
        train_cross_entropy, valid_cross_entropy = np.average(train_cross_entropies), np.average(valid_cross_entropies)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_train_cross_entropies.append(train_cross_entropy)
        avg_valid_cross_entropies.append(valid_cross_entropy)
        print('Train Epoch: [{}/{}]  train_loss: {:.6f} valid_loss: {:.6f} train_CrossEntropy: {:.6f} '
              'valid_CrossEntropy: {:.6f} Curvature: {:.6f}'.format(epoch, epochs, train_loss, valid_loss,
                                                                    train_cross_entropy, valid_cross_entropy,
                                                                    curvature))
        # clear lists to track next epoch
        train_losses, valid_losses, train_cross_entropies, valid_cross_entropies = [], [], [], []
        torch.save(net.module.state_dict(), result_path + '/epoch_{}.pth'.format(epoch))
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        '''e = time.time()
        print("1 epoch time:{:.0f}s".format(e - s))'''

    # load the last checkpoint with the best model
    if CFG.gpu:
        net.module.load_state_dict(torch.load(result_path + '/checkpoint.pt'))
    else:
        net.load_state_dict(torch.load(result_path + '/checkpoint.pt'))
    return avg_train_losses, avg_valid_losses, avg_train_cross_entropies, avg_valid_cross_entropies


def test(net):
    net.eval()
    test_loss = []
    curvature = 0
    crossentropy = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = net(data).to(device)
        loss, cr, cu = my_loss_function(output, target,CFG.l)
        test_loss.append(loss.item())
        crossentropy += cr
        curvature += cu
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss = np.average(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    crossentropy = crossentropy / len(test_loader)
    curvature = curvature / len(test_loader)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                              len(test_loader.dataset), accuracy))
    return test_loss, crossentropy, accuracy


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, batch_idx=None, epoch=None):
        cross_entropy = CrossEntropy(x, y)
        if CFG.l == 1.0:
            curvature = torch.tensor(0)
        else:
            # calculate first order derivative of all weights
            first_grad = torch.autograd.grad(cross_entropy, net.parameters(), create_graph=True, retain_graph=True)
            if CFG.topk == 1.0:
                hesse = hessian(first_grad, net, device)
                curvature = torch.sum(torch.pow(hesse, 2))
                if batch_idx==len(train_loader):
                    plot_hesse(hesse, epoch, result_path)
            else:
                curvature = hessian_random_topk(first_grad, net, CFG.topk)
        return cross_entropy * CFG.l + curvature * (1 - CFG.l), cross_entropy.item(), curvature.item()


class Logger(object):
    def __init__(self, filename=result_path + '/train.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    sys.stdout = Logger(stream=sys.stdout)
    print(CFG)
    device = setup(gpu=CFG.gpu)
    train_loader, test_loader, valid_loader = getData(CFG.dataset, CFG.batch_size)
    net = getModel(CFG.model, device)
    if CFG.gpu:
        net = nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(), CFG.learning_rate, weight_decay=0.0001)
    my_loss_function = MyLoss()
    CrossEntropy = nn.CrossEntropyLoss()
    train_loss, valid_loss, train_cross_entropy, valid_cross_entropy = train(net, CFG.epoch)
    test(net)
    plot_loss(train_loss, valid_loss, train_cross_entropy, valid_cross_entropy, result_path=result_path)
    if CFG.gpu:
        torch.save(net.module.state_dict(), result_path + '/{}_{}_{}.pth'.format(CFG.dataset, CFG.model, CFG.l))
    else:
        torch.save(net.state_dict(), result_path + '/{}_{}_{}.pth'.format(CFG.dataset, CFG.model, CFG.l))
