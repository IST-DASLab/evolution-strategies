import random
import copy
from time import time

import torch
import numpy as np
import torch.multiprocessing as mp

TIME = False

RANK = 0

def id(*args, **kwargs):
    return args[0]


def thread_loop(in_queue,
                out_queue,
                model,
                args):
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
    random.seed(args['seed'])
    global RANK
    RANK = args['rank']

    quantize = args.get('quantize', id)
    noise_model = args['noise_model']
    evaluate = args['evaluate']
    get_next_model_child = args['get_next_model_child']
    perturbs_per_process = args['perturbs_per_process']

    for _ in range(args['num_epochs']):
        # t0 = time()
        args['start_barrier'].wait()
        # t1 = time()
        # t2 = 0.0
        # t3 = 0.0
        # t4 = 0.0
        # t5 = 0.0
        for _ in range(perturbs_per_process):
            s = int(random.random() * (2**32))
            # _t0 = time()
            model_n = noise_model(copy.deepcopy(model), s)
            # _t1 = time()
            model_q = quantize(model_n)
            # _t2 = time()
            score = evaluate(model_q)
            # _t3 = time()
            out_queue.put((score, s))
            # _t4 = time()
            # t2 += _t1 - _t0
            # t3 += _t2 - _t1
            # t4 += _t3 - _t2
            # t5 += _t4 - _t3
        # t6 = time()
        model = in_queue.get()
        # t7 = time()
        args['end_barrier'].wait()
        # t8 = time()
        # if RANK == 1:
        #     print('start wait   {:2.6f}'.format(t1 - t0))
        #     print('noise model  {:2.6f}'.format(t2))
        #     print('quantize     {:2.6f}'.format(t3))
        #     print('evaluate     {:2.6f}'.format(t4))
        #     print('out Q put    {:2.6f}'.format(t5))
        #     print('in Q get     {:2.6f}'.format(t7 - t6))
        #     print('end wait     {:2.6f}'.format(t8 - t7))



def main_loop(args, debug=False, seed=None):
    '''
    The main loop of the program. This is where we spawn processes, and do everything.
    On our side, the main loop looks like this:

    >>> for _ in range(N):
    >>>     results = get_all_results()
    >>>     results.sort()
    >>>     broadcast(results[:k])

    Somehow we must also shut down the threads?
    '''

    TIME = debug
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch.set_num_threads(1)

    num_epochs = args['num_epochs']
    num_processes = args['num_processes']
    perturbs_per_process = args['perturbs_per_process']

    args['start_barrier'] = mp.Barrier(num_processes)
    args['end_barrier'] = mp.Barrier(num_processes)

    model_queue = mp.Queue()
    score_queue = mp.Queue()
    model = args['get_model']()

    get_next_model_main = args['get_next_model_main']
    get_next_model_child = args['get_next_model_child']
    save_model = args.get('save_model', None)

    # Start all processes
    processes = []
    for rank in range(num_processes):
        args['seed'] = int(random.random() * (2**32))
        args['rank'] = rank
        p = mp.Process(target=thread_loop, args=(model_queue, score_queue, model, args))
        p.start()
        processes.append(p)

    start_time = time()
    for epoch in range(num_epochs):
        t0 = time()
        results = [score_queue.get() for _ in range(num_processes * perturbs_per_process)]
        t1 = time()
        results.sort() # sort by accuracy
        t2 = time()
        # print(results[:5])
        model_builder = get_next_model_main([(score, seed) for ((score, _), seed) in results])
        t3 = time()
        model = get_next_model_child(model, model_builder)
        t4 = time()
        for _ in range(num_processes):
            model_queue.put(model)
        t4 = time()
        if save_model:
            save_model(model, model_save_path)
        t5 = time()
        print('epoch: {:4}; time: {:5.0f}; best score: {:1.5f}; accuracy: {:1.5f}'.format(
            epoch, time() - start_time, results[0][0][0], results[0][0][1]))
        # print('[main] get results   {:2.6f}'.format(t1 - t0))
        # print('[main] sort          {:2.6f}'.format(t2 - t1))
        # print('[main] getnextmodel  {:2.6f}'.format(t3 - t2))
        # print('[main] put to Q      {:2.6f}'.format(t4 - t3))
        # print('[main] get and save  {:2.6f}'.format(t5 - t4))

    # We are done.
    for p in processes:
        p.join()
