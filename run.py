from random import seed, random
from sys import argv
from time import time
import pickle
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

# Data saving stuff
model_save_path = './model'

# Parameters
lr = 1e-4
sigma = 0.05
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
        rands = np.random.normal(0, 1, w.size())
        w.data += torch.from_numpy(rands * sigma).float()
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
    N = len(trainloader)
    for (i, data) in enumerate(testloader, 1):
        inputs, _labels = data
        outputs = model(torch.autograd.Variable(inputs)).data
        labels = torch.Tensor([one_hot(int(i), 10) for i in _labels])
        error += (outputs - labels).pow(2).sum()
    return error / batch_size / N


class Net(nn.Module):
    '''The model. This needs to be stored at the top level of the file, so that
    we can serialize it.'''
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


def make_model():
    '''Construct the initial pytorch model'''
    return Net()


def get_next_model_main(results):
    '''
    results: [(score, seed), ...]
    It is sorted. Whatever is returned here is sent to all children and given to
    `get_next_model_child()`
    '''
    [scores, seeds] = map(list, zip(*results))
    norm = lambda s: (s - scores[0]) / (scores[-1] - scores[0])
    normalized_scores = [(1 - norm(s), i) for (i, s) in enumerate(scores)]
    s = sum(map(lambda t: t[0], normalized_scores))
    return [(score / s, seeds[i]) for (score, i) in normalized_scores]


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
            update += lr / (batch_size * sigma) * frac * rands
        asd = torch.from_numpy(update).float()
        param += asd
    return model


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

    for _ in range(args['num_epochs']):
        s = int(random() * (2**32))
        args['start_barrier'].wait()
        model_n = noise_model(copy.deepcopy(model), s)
        model_q = quantize(model_n)
        score = evaluate(model_q)
        out_queue.put((score, s))
        reconstruct_data = in_queue.get()
        model = get_next_model_child(model, reconstruct_data)
        # print('reconstructed model score: ', evaluate(model))
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
    np.random.seed(0)
    torch.manual_seed(0)

    torch.set_num_threads(1)
    num_epochs = 5 # 100
    n_processes = int(argv[1]) # mp.cpu_count()
    args = {
        'num_epochs': num_epochs,
        'start_barrier': mp.Barrier(n_processes),
        'end_barrier': mp.Barrier(n_processes),
    }

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

    start_time = time()
    for _epoch in range(num_epochs):
        results = [score_queue.get() for _ in range(n_processes)]
        results.sort()
        model_builder = get_next_model_main(results)
        for _ in range(n_processes):
            model_queue.put(model_builder)
        print(results)
        print('[{:4.1f}] best score: {}'.format(time() - start_time, results[0][0]))
        model = get_next_model_child(model, model_builder)
        save_model(model, model_save_path)

    # We are done.
    for p in processes:
        p.join()


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
    main_loop()
