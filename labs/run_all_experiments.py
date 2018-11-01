import concurrent
import os
import pickle
import time

from data.utils import load_data_seeds, store_data_seeds
from experiment.factory import create_run_experiment
from experiment.utils import chdir_data_set
from flags import FLAGS
from data.registry import create_data_loader
import numpy as np

# python -m run_all_experiments -dir /home/pawel/PycharmProjects/RM_labs/nn_pdf_ar -models nn_pdf_ar -per_process_gpu_memory_fraction 0.01 -num_workers 2


def run(model, data_set, data_sample_random_seeds):
    chdir_data_set(data_set)

    FLAGS.data_set=data_set
    FLAGS.model = model

    print("Running experiment with FLAGS: %s" %FLAGS.flag_values_dict())

    exp = create_run_experiment()
    if exp is not None and FLAGS.num_samples_best_eval > 0:
        exp.train_eval_best_model(data_sample_random_seeds)


def run_true_metrics(data_set):
    chdir_data_set(data_set)

    FLAGS.data_set = data_set

    data_loader = create_data_loader()
    if data_loader.can_compute_ll():

        train_ll = data_loader.ll(data_loader.train_x, data_loader.train_y)
        validation_ll=data_loader.ll(data_loader.validation_x, data_loader.validation_y)
        test_ll=data_loader.ll(data_loader.test_x, data_loader.test_y)

        with open('real_metrics.pkl', 'wb') as f:
            data = {}
            data["train_ll"] = train_ll
            data["validation_ll"] = validation_ll
            data["test_ll"] = test_ll

            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    from asynch import Callable
    from models.utils import printProgressBar, generate_seed

    printProgressBar(0, 100, prefix='Progress run all:', suffix='Complete')

    funcs = []
    print("Running all experiments for models: {models} and data sets: {data_sets}".format(models=FLAGS.models, data_sets=FLAGS.data_sets))

    loaded_data_seeds = load_data_seeds()

    if FLAGS.data_sample_random_seeds != "":
        print("data seeds from command line")
        data_sample_random_seeds = [int(seed) for seed in FLAGS.data_sample_random_seeds.split(",")]
    elif loaded_data_seeds is not None:
        print("loaded data seeds")
        data_sample_random_seeds = loaded_data_seeds
    else:
        print("generating data seeds")
        data_sample_random_seeds = [generate_seed() for _ in range(FLAGS.num_samples_best_eval)]


    store_data_seeds(data_sample_random_seeds)
    print("Data sample seeds: %s"%data_sample_random_seeds)

    for data_set in [data_set.strip() for data_set in FLAGS.data_sets.split(",")]:
        funcs.append(Callable(run_true_metrics, data_set=data_set))
        for model in [model.strip() for model in FLAGS.models.split(",")]:
            funcs.append(Callable(run,model=model, data_set=data_set,data_sample_random_seeds = data_sample_random_seeds))

    # First prefetch all data
    for data_set in FLAGS.data_sets.split(","):
        FLAGS.data_set = data_set
        target_dir = os.path.join(FLAGS.dir, data_set)
        os.makedirs(target_dir, exist_ok=True)
        os.chdir(target_dir)
        data_loader = create_data_loader()

    done = 0.0
    futures = []
    res = [None] * len(funcs)
    with concurrent.futures.ProcessPoolExecutor(FLAGS.num_parallel_experiments) as executor:
        for i, fun in enumerate(funcs):
            inserted = False
            while not inserted:
                if len(futures) < FLAGS.num_parallel_experiments:
                    futures.append((i, executor.submit(fun)))
                    inserted = True

                for fut in list(futures):
                    try:
                        res[fut[0]] = fut[1].result(0)
                        done += 1
                        printProgressBar(done / len(funcs)*100, 100, prefix='Progress run all:', suffix='Complete')
                        futures.remove(fut)
                    except concurrent.futures.TimeoutError:
                        pass

                if len(futures) == FLAGS.num_parallel_experiments:
                    time.sleep(1)

    for fut in list(futures):
        res[fut[0]] = fut[1].result()
        done += 1
        printProgressBar(done / len(funcs) * 100, total=100, prefix='Progress run all:', suffix='Complete')