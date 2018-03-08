import random
import math
from time import time
import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import base

LR = 0.001
SIGMA = 0.002
batch_size = 1

transform = transforms.Compose(
    [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.lin = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = self.lin(x)
        return F.relu(x)


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.conv2 = nn.Conv2d(4, 8, 5, padding=2)
        self.lin1 = nn.Linear(7 * 7 * 8, 1024)
        self.lin2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 8)
        x = self.lin1(x)
        x = self.lin2(x)
        return x


class MNIST(base.ES):

    def __init__(self, model):
        super(MNIST, self).__init__(model)
        self.epoch_count = 0;
        self.previous_time = time()
        self.start_time = self.previous_time

    def perturb(self, model, seed):
        np.random.seed(seed)
        for w in model.parameters():
            rands = np.random.normal(0, SIGMA, w.size())
            noise = torch.from_numpy(rands).float()
            w.data += noise
        return model

    def evaluate(self, model):
        def one_hot(i, n):
            l = [0.0] * n
            l[i] = 1.0
            return l

        error = 0.0
        N = len(trainloader)
        for (i, data) in enumerate(trainloader, 1):
            inputs, _labels = data
            Variable = torch.autograd.Variable
            outputs = model(Variable(inputs))
            ce = F.cross_entropy(outputs, Variable(_labels))
            error += float(ce)
        return error / batch_size / N

    def post_perturb(self, model):
        # TODO: quantize here
        return model

    def aggregate_results(self, results):
        '''
        results: [(evaluate(model), seed), ...]
        It is sorted. Whatever is returned here is sent to all children and given to
        `reconstruct()`
        '''
        if results[-1][0] == results[0][0]:
            return [(0, s) for (n, s) in results]
        normalize = lambda n: 1 - ((n - results[0][0]) / (results[-1][0] - results[0][0]))
        _sum = sum(normalize(n) for (n, _) in results)
        return [(normalize(n) / _sum, s) for (n, s) in results]

    def reconstruct(self, model, results):
        '''
        `model` is the current full precision model.
        `results` is whatever aggregate_results() returned.
        '''
        updates = [torch.Tensor(w.size()).zero_() for w in model.parameters()]

        for (frac, seed) in results:
            np.random.seed(seed)
            for (i, w) in enumerate(model.parameters()):
                rands = np.random.normal(0, SIGMA, w.size())
                noise = torch.from_numpy(rands).float() * frac
                updates[i] += noise
        N = len(results)
        factor = 1 / (N * SIGMA)
        for (w, u) in zip(model.parameters(), updates):
            w.data += u * factor
        return model

    def post_epoch(self, results, aggr):
        def calc_precision():
            error = 0.0
            N = len(testloader)
            correct = 0
            for (i, data) in enumerate(testloader, 1):
                inputs, _labels = data
                Variable = torch.autograd.Variable
                outputs = self.model(Variable(inputs))

                ce = F.cross_entropy(outputs, Variable(_labels))
                error += float(ce)

                pred = outputs.max(1, keepdim=True)[1]
                target = torch.autograd.Variable(_labels.view_as(pred))
                correct += int(pred.eq(target).sum())
            return correct / batch_size / N
        t0 = time()
        self.model = self.reconstruct(self.model, aggr)
        prec = calc_precision()
        t1 = time()
        print('epoch: {:4}; time: {:5.0f} ({:4.0f}ms; overhead {:4.0f}ms); best score: {:1.5f}; precision: {:1.5f}'.format(
            self.epoch_count,
            t1 - self.start_time,
            (t0 - self.previous_time) * 1000,
            (t1 - t0) * 1000,
            results[0][0],
            prec))
        self.previous_time = t1
        self.epoch_count += 1


# for reproducability
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

model = MNIST(Net())

# import pickle
# for INDEX in range(1, 6):
#     with open('mnist-large-{}'.format(INDEX), 'rb') as f:
#         net = pickle.load(f)
#
#     model = MNIST(net).model
#
#     error = 0.0
#     N = len(testloader)
#     correct = 0
#     for (i, data) in enumerate(testloader, 1):
#         inputs, _labels = data
#         Variable = torch.autograd.Variable
#         outputs = model(Variable(inputs))
#
#         ce = F.cross_entropy(outputs, Variable(_labels))
#         error += float(ce)
#
#         pred = outputs.max(1, keepdim=True)[1]
#         target = torch.autograd.Variable(_labels.view_as(pred))
#         correct += int(pred.eq(target).sum())
#     print('i={}, precision={}, error={}'.format(INDEX, correct / batch_size / N, error))
# import sys
# sys.exit(0)

def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+6.3f} x^{} '.format(w, len(W) - i)
    result += '{:+6.3f}'.format(b[0])
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ES')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--sigma', type=float, default=0.002, metavar='SD',
                        help='noise standard deviation')
    parser.add_argument('-n', type=int, default=2, metavar='processes',
                        help='number of processes')
    parser.add_argument('-e', type=int, default=1000, metavar='epochs',
                        help='number of epochs')
    parser.add_argument('-p', type=int, default=10000, metavar='population',
                        help='population size')
    args = parser.parse_args()

    LR = args.lr
    SIGMA = args.sigma

    base.main_loop(model,
                   num_epochs=args.e,
                   offspring_per_process=args.p // args.n,
                   num_processes=args.n,
                   seed=SEED)
