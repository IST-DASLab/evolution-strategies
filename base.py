import random
import copy
from time import time

import torch
import numpy as np
import torch.multiprocessing as mp


class ES():
    def __init__(self, model):
        self.model = model
        self.seed = int(random.random() * (2 ** 32))
        self.debug = False

    def perturb(self, model, seed):
        raise NotImplementedError('Please supply an `perturb` method when subclassing ES')

    def evaluate(self, model):
        raise NotImplementedError('Please supply an `evaluate` method when subclassing ES')

    def pre_perturb(self, model):
        return model

    def post_perturb(self, model):
        return model

    def aggregate_results(results):
        '''Take a list of results from `evaluate()`, and transform it somehow.
        The output of this function is later passed to `reconstruct`.'''
        pass

    def reconstruct(self, model, aggr):
        raise NotImplementedError('Please supply a `reconstruct` method when subclassing ES')


def thread_loop(
        es,
        in_queue,
        out_queue,
        start_barrier,
        end_barrier,
        init_seed,
        num_epochs=100,
        offspring_per_process=100):
    '''
    This is what each thread performs. It works something like this:

    >>> while True:
    >>>     model = noise_model(model)
    >>>     score = evaluate(model_q)
    >>>     broadcast(score)
    >>>     k_best = get_k_best()
    >>>     model = avg(k_best)

    '''
    random.seed(init_seed)

    for _ in range(num_epochs):
        start_barrier.wait()
        for _ in range(offspring_per_process):
            s = int(random.random() * (2**32))
            model_ = copy.deepcopy(es.model)
            pre = es.pre_perturb(model_)
            perturbed = es.perturb(pre, s)
            post = es.post_perturb(perturbed)
            score = es.evaluate(post)
            out_queue.put((score, s))
        aggr = in_queue.get()
        es.model = es.reconstruct(es.model, aggr)
        end_barrier.wait()


def main_loop(
        es,
        num_epochs=100,
        offspring_per_process=100,
        num_processes=1,
        seed=None,
        save_model=None):
    '''
    The main loop of the program. This is where we spawn processes, and do everything.
    On our side, the main loop looks like this:

    >>> for _ in range(N):
    >>>     results = get_all_results()
    >>>     results.sort()
    >>>     broadcast(results[:k])

    Somehow we must also shut down the threads?
    '''

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch.set_num_threads(1)

    start_barrier = mp.Barrier(num_processes)
    end_barrier = mp.Barrier(num_processes)

    model_queue = mp.Queue()
    score_queue = mp.Queue()

    # Start all processes
    processes = []
    for rank in range(num_processes):
        es.debug = rank == 0
        p = mp.Process(target=thread_loop, args=(
            es,
            model_queue,
            score_queue,
            start_barrier,
            end_barrier,
            int(random.random() * (2**32))),
            kwargs={
                'num_epochs': num_epochs,
                'offspring_per_process': offspring_per_process,
            })
        p.start()
        processes.append(p)

    start_time = time()
    for epoch in range(num_epochs):
        results = [score_queue.get() for _ in range(num_processes * offspring_per_process)]

        aggr = es.aggregate_results(results)
        for _ in range(num_processes):
            model_queue.put(aggr)

        if save_model:
            save_model(model, model_save_path)
        print('epoch: {:4}; time: {:5.0f}; best score: {:1.5f}'.format(
            epoch, time() - start_time, results[0][0]))

    # We are done.
    for p in processes:
        p.join()
