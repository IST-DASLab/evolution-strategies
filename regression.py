import random
import math
from time import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import base


class Regression(base.ES):

    def __init__(self, model):
        super(Regression, self).__init__(model)
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
        def make_features(x):
            """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
            x = x.unsqueeze(1)
            return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

        def get_batch(batch_size):
            def f(x):
                """Approximated function."""
                return x.mm(W_target) + b_target[0]

            """Builds a batch i.e. (x, f(x)) pair."""
            random_batch = torch.randn(batch_size)
            x = make_features(random_batch)
            y = f(x)
            return Variable(x), Variable(y)

        data, real = get_batch(batch_size)
        output = model(data)
        err = float(F.smooth_l1_loss(output, real).data[0])
        return err / batch_size

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
        if self.debug:
            print()
        for param in model.state_dict().values():
            N = param.size()
            update = np.zeros(N)
            for (frac, seed) in results:
                np.random.seed(seed)
                rands = np.random.normal(0, 1, N)
                update += frac * rands
            update *= 1 / (len(results) * SIGMA ** 2)
            param += torch.from_numpy(update * LR).float()
        return model

    def post_epoch(self, results, aggr):
        t = time()
        self.model = self.reconstruct(self.model, aggr)
        print('Learned function: ' + poly_desc(self.model.weight.data.view(-1),
                                               self.model.bias.data))
        print('Actual function:  ' + poly_desc(W_target.view(-1), b_target))
        print('epoch: {:4}; time: {:5.0f} ({:4.0f}ms); best score: {:1.5f}'.format(
            self.epoch_count,
            t - self.start_time,
            (t - self.previous_time) * 1000,
            results[0][0]))
        self.previous_time = t
        self.epoch_count += 1


LR = 0.001
SIGMA = 0.002

POLY_DEGREE = 6
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


model = Regression(torch.nn.Linear(W_target.size(0), 1))
batch_size = 32


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
                   seed=0)
