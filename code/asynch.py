import concurrent.futures
import time
from concurrent.futures import Executor

from IPython.core.display import display
from ipywidgets import FloatProgress

from flags import FLAGS


class SameProcessFuture:
    def __init__(self, res):
        self.res = res

    def result(self, timeout=None):
        return self.res


class SameProcessExecutor(Executor):

    def submit(self, fn, *args, **kwargs):
        return SameProcessFuture(fn(*args, **kwargs))


class WorkItem:
    def __init__(self, id, args_list, args_named, future):
        self.args_list = args_list
        self.args_named = args_named
        self.future = future
        self.id = id

    def __eq__(self, other):
        return self.id == other.id


class Callable:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*self.args, **self.kwargs)

def invoke_in_process_pool(num_workers, *funcs):
    if FLAGS.plot:
        progress = FloatProgress(min=0, max=1)
        display(progress)

    done = 0.0
    futures = []
    res = [None] * len(funcs)
    with SameProcessExecutor() if num_workers <= 0 else concurrent.futures.ProcessPoolExecutor(
            num_workers) as executor:
        for i, fun in enumerate(funcs):
            inserted = False
            while not inserted:
                if len(futures) < num_workers:
                    futures.append((i, executor.submit(fun)))
                    inserted = True

                for fut in list(futures):
                    try:
                        res[fut[0]] = fut[1].result(0)
                        done += 1
                        if FLAGS.plot:
                            progress.value = done / len(funcs)
                        futures.remove(fut)
                    except concurrent.futures.TimeoutError:
                        pass

                if len(futures) == num_workers:
                    time.sleep(1)

    for fut in list(futures):
        res[fut[0]] = fut[1].result()
        done += 1
        if FLAGS.plot:
            progress.value = done / len(funcs)

    return res