from random import seed, random
from sys import argv
from time import time


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

lr = 1e-4


def noise_model(model, seed):
    np.random.seed(seed)
    for w in model.parameters():
        w.data += (np.random.normal(0, 1) * lr)
    return model


def quantize(model):
    return model


def evaluate(model):
    def one_hot(i, n):
        l = [0.0] * n
        l[i] = 1.0
        return l
    error = 0.0
    N = len(trainloader)
    for (i, data) in enumerate(trainloader, 1):
        # TODO: use whole data set
        if i > 2000: break;
        inputs, _labels = data
        outputs = model(torch.autograd.Variable(inputs)).data
        labels = torch.Tensor([one_hot(int(i), 10) for i in _labels])
        error += (outputs - labels).abs().sum()
    return error


def make_model():
    '''Construct the initial pytorch model'''
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 12, 5)
            self.lin = nn.Linear(24 * 24 * 12, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = x.view(-1, 24 * 24 * 12)
            x = self.lin(x)
            return x

    return Net()


def get_next_model_main(results):
    '''
    results: [(score, seed), ...]
    It is sorted. Whatever is returned here is sent to all children and given to
    `get_next_model_child()`
    '''
    return results[0][1]

def get_next_model_child(model, results):
    '''
    model is the current full precision model.
    results is whatever get_next_model_main() returned.'''
    return noise_model(model, results)


def thread_loop(in_queue, out_queue, model, args):
    '''
    This is what each thread performs. It works something like this:

    >>> while True:
    >>>     model = noise_model(model)
    >>>     model_q = quantize(model)
    >>>     score = evaluate(model_q)
    >>>     broadcast(score)
    >>>     k_best = get_k_best()
    >>>     model = avg(k_best)

    '''
    seed(args['seed'])

    for _ in range(args['num_iters']):
        s = int(random() * (2**32))
        args['start_barrier'].wait()
        model_n = noise_model(model, s)
        model_q = quantize(model_n)
        score = evaluate(model_q)
        out_queue.put((score, s))
        reconstruct_data = in_queue.get()
        model = get_next_model_child(model, reconstruct_data)
        args['end_barrier'].wait()



def main_loop():
    '''
    The main loop of the program. This is where we spawn processes, and do everything.
    On our side, the main loop looks like this:

    >>> for _ in range(N):
    >>>     results = get_all_results()
    >>>     results.sort()
    >>>     broadcast(results[:k])

    Somehow we must also shut down the threads?
    '''

    # TODO: remove deterministic seed
    seed(0)

    torch.set_num_threads(1)
    num_iters = 5 # 100
    n_processes = int(argv[1]) # mp.cpu_count()
    args = {
        'num_iters': num_iters,
        'start_barrier': mp.Barrier(n_processes),
        'end_barrier': mp.Barrier(n_processes),
    }

    print('each process uses %d threads' % torch.get_num_threads())

    model_queue = mp.Queue()
    score_queue = mp.Queue()
    model = make_model()
    # Start all processes
    processes = []
    for rank in range(n_processes):
        args['seed'] = int(random() * (2**32))
        args['rank'] = rank
        p = mp.Process(target=thread_loop, args=(model_queue, score_queue, model, args))
        p.start()
        processes.append(p)

    # Get scores, find best, and report back.
    t = 0.0
    for i in range(num_iters):
        t0 = time()
        results = [score_queue.get() for _ in range(n_processes)]
        results.sort()
        ret = get_next_model_main(results)
        for _ in range(n_processes):
            model_queue.put(ret)
        t1 = time()
        t += t1 - t0
    t /= num_iters
    print('Evaluated {} models in {:4.2f} secs ({} ms/m)'.format(
        n_processes, t, t / n_processes * 1000))

    # We are done.
    for p in processes:
        p.join()


if __name__ == '__main__':
    main_loop()
