from random import seed, random
from sys import argv
from time import time
import math
import pickle
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

import base

# Data saving stuff
model_save_path = './mnist-model'

# Parameters
lr = 0.001
sigma = 0.01
batch_size = 1

transform = transforms.Compose(
    [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


def noise_model(model, seed):
    '''
    Perturb the model with the given `seed`.
    '''
    np.random.seed(seed)
    for w in model.parameters():
        rands = np.random.normal(0, 1, w.size()) * sigma
        w.data += torch.from_numpy(rands).float()
    return model


def quantize(model):
    '''
    Quantize the model.  It is the quantized copy that gets fed into `evaluate()`.
    '''
    return model


def evaluate(model):
    '''
    Run through the test images, and see how well we're doing. Return our score. Lower is better.
    '''
    def one_hot(i, n):
        l = [0.0] * n
        l[i] = 1.0
        return l
    error = 0.0
    N = len(testloader)
    correct = 0
    for (i, data) in enumerate(testloader, 1):
        inputs, _labels = data
        Variable = torch.autograd.Variable
        outputs = model(Variable(inputs))

        ce = F.cross_entropy(outputs, Variable(_labels))
        error += float(ce)

        pred = outputs.max(1, keepdim=True)[1]
        target = torch.autograd.Variable(_labels.view_as(pred))
        correct += int(pred.eq(target).sum())
    return (error / batch_size / N, correct / batch_size / N)


class Net(nn.Module):
    '''The model. This needs to be stored at the top level of the file, so that
    we can serialize it.'''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.conv2 = nn.Conv2d(4, 8, 5, padding=2)
        self.lin = nn.Linear(7 * 7 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 8)
        x = self.lin(x)
        return F.relu(x)


def make_model():
    '''Construct the initial pytorch model'''
    return Net()


def get_next_model_main(results):
    '''
    results: [(score, seed), ...]
    It is sorted. Whatever is returned here is sent to all children and given to
    `get_next_model_child()`
    '''
    normalize = lambda n: 1 - ((n - results[0][0]) / (results[-1][0] - results[0][0]))
    _sum = sum(normalize(n) for (n, _) in results)
    return [(normalize(n) / _sum, s) for (n, s) in results]


def get_next_model_child(model, results):
    '''
    `model` is the current full precision model.
    `results` is whatever get_next_model_main() returned.
    '''
    for param in model.state_dict().values():
        N = param.size()
        update = np.zeros(N)
        for (frac, seed) in results:
            np.random.seed(seed)
            rands = np.random.normal(0, 1, N)
            update += frac * rands
        param += torch.from_numpy(update * lr).float()
    return model


def save_model(model, path):
    '''Saves the current model to the file specified'''
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    '''Load the model form the specified file'''
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # TODO: add argument handling, so that we can pass sigma, lr, and whether to train a new model
    # or load an old one.
    args = {
            'get_model': make_model,
            'noise_model': noise_model,
            'quantize': quantize,
            'evaluate': evaluate,
            'get_next_model_main': get_next_model_main,
            'get_next_model_child': get_next_model_child,
            'num_epochs': 100,
            'num_processes': 4,
            'perturbs_per_process': 2,
    }
    base.main_loop(args)
