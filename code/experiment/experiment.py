import concurrent.futures
import time

from IPython.core.display import display
from ipywidgets import FloatProgress
from skopt import Optimizer

from asynch import WorkItem, Callable, invoke_in_process_pool, SameProcessExecutor
from experiment.train_eval import TrainEvalModelFactory
import os

from experiment.utils import load_best_model_exp
from flags import FLAGS
from models.utils import load_model, generate_seed, save_best_model_exp, printProgressBar, predict_estimator
import numpy as np


class Runner:

    def __init__(self, num_workers=None, num_samples=None, num_samples_best_eval=None, space=None, train_eval_model: TrainEvalModelFactory=None,
                 params_converter=None):
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.num_samples_best_eval = num_samples_best_eval

        self.optimizer = Optimizer(
            dimensions=space,
            random_state=1
        )
        self.res = None
        self.params_converter = params_converter
        self.train_eval_model = train_eval_model
        self.space = space

        self.best_model_loss = None
        self.best_model_params = None
        self.best_model_dir = None
        self.best_model = None

        os.makedirs('tensorboard', exist_ok=True)

        self.best_model_train_eval = {}
        self.best_model_validation_eval = {}
        self.best_model_test_eval = {}

        self.best_model_train_ll = None
        self.best_model_valid_ll = None
        self.best_model_test_ll = None

    def load_best_model(self):
        loaded = load_best_model_exp(self.train_eval_model.model_name)
        self.__dict__.update(loaded)
        self.best_model = load_model(self.best_model_dir, self.best_model_params)
        print("Loaded experiment with best model: {model} for data set: {data_set}".format(model=FLAGS.model,
                                                                                           data_set=FLAGS.data_set))

    def load_best_model_or_run(self):
        try:
            self.load_best_model()
        except:
            print(
                "Running experiment model: {model} for data set: {data_set}".format(model=FLAGS.model, data_set=FLAGS.data_set))
            self.run()

    def to_named_params(self, x):
        args = {}
        for i, item in enumerate(self.space):
            if isinstance(x[i], np.int64):
                x[i] = int(x[i])
            args[item.name] = x[i]

        args['tensorboard_folder'] = os.path.join("tensorboard")
        args['tf_random_seed'] = generate_seed()
        args['x_size'] = self.train_eval_model.data_loader.train_x.shape[1]
        args['y_size'] = self.train_eval_model.data_loader.train_y.shape[1]

        return self.params_converter(args) if self.params_converter is not None else args

    def train_eval_task_finished(self, futures, wi, model_dir, train_eval, validation_eval, test_eval):
        if validation_eval is None:
            futures.remove(wi)
            return
        model_loss = validation_eval["loss"]
        self.res = self.optimizer.tell(wi.args_list, model_loss);
        futures.remove(wi)

        if self.best_model_loss is None or self.best_model_loss > model_loss:
            self.best_model_loss = model_loss
            self.best_model_dir = model_dir
            self.best_model_params = wi.args_named

            self.save()

    def save(self):
        save_best_model_exp(self.train_eval_model.model_name, self)

    def best_task_eval_finished(self, futures, wi, model_dir, train_eval, validation_eval, test_eval):
        futures.remove(wi)
        self.best_model_train_eval[wi.id] = train_eval
        self.best_model_validation_eval[wi.id] = validation_eval
        self.best_model_test_eval[wi.id] = test_eval

    def run(self):
        futures = []
        if FLAGS.plot:
            progress = FloatProgress(min=0, max=1)
            display(progress)
        else:
            printProgressBar(0, self.num_samples, prefix='Progress experiment {model}/{data_set}:'.
                             format(model=FLAGS.model, data_set=FLAGS.data_set), suffix='Complete', length=50)

        done = 0.0
        with (SameProcessExecutor() if self.num_workers <= 0 else concurrent.futures.ProcessPoolExecutor(
                self.num_workers)) as executor:
            for i in range(self.num_samples):
                inserted = False
                while not inserted:
                    if len(futures) < self.num_workers or self.num_workers <= 0:
                        x = self.optimizer.ask()  # x is a list of n_points points
                        objective_fun = self.train_eval_model.create_train_eval(i)
                        args_named = self.to_named_params(x)
                        futures.append(
                            WorkItem(i, x, args_named, executor.submit(objective_fun, args=None, **args_named)))
                        inserted = True

                    for wi in list(futures):
                        try:
                            model_dir, train_eval, validation_eval, test_eval = wi.future.result(0)
                            self.train_eval_task_finished(futures, wi, model_dir, train_eval, validation_eval,
                                                          test_eval)
                            done += 1
                            if FLAGS.plot:
                                progress.value = done / self.num_samples
                            else:
                                printProgressBar(done, self.num_samples, prefix='Progress experiment {model}/{data_set}:'.
                             format(model=FLAGS.model, data_set=FLAGS.data_set), suffix='Complete', length=50)
                        except concurrent.futures.TimeoutError:
                            pass

                    if len(futures) != 0 and len(futures) == self.num_workers:
                        time.sleep(1)

        for wi in list(futures):
            model_dir, train_eval, validation_eval, test_eval = wi.future.result()
            self.train_eval_task_finished(futures, wi, model_dir, train_eval, validation_eval, test_eval)
            done += 1
            if FLAGS.plot:
                progress.value = done / self.num_samples
            else:
                printProgressBar(done, self.num_samples, prefix='Progress experiment {model}/{data_set}:'.
                             format(model=FLAGS.model, data_set=FLAGS.data_set), suffix='Complete', length=50)

        self.best_model = load_model(self.best_model_dir, self.best_model_params)

        predict_train, predict_valid, predict_test = invoke_in_process_pool(self.num_workers,
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.train_x,
                                                                                     self.train_eval_model.data_loader.train_y),
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.validation_x,
                                                                                     self.train_eval_model.data_loader.validation_y),
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.test_x,
                                                                                     self.train_eval_model.data_loader.test_y)
                                                                            )

        self.best_model_train_ll = predict_train["log_likelihood"]
        self.best_model_valid_ll = predict_valid["log_likelihood"]
        self.best_model_test_ll = predict_test["log_likelihood"]

        self.save()

    def eval_best_model_and_save(self):
        predict_train, predict_valid, predict_test = invoke_in_process_pool(self.num_workers,
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.train_x,
                                                                                     self.train_eval_model.data_loader.train_y),
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.validation_x,
                                                                                     self.train_eval_model.data_loader.validation_y),
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.test_x,
                                                                                     self.train_eval_model.data_loader.test_y)
                                                                            )

        self.best_model_train_ll = predict_train["log_likelihood"]
        self.best_model_valid_ll = predict_valid["log_likelihood"]
        self.best_model_test_ll = predict_test["log_likelihood"]

        self.save()


    def __str__(self):
        print("Best score=%.4f" % min(self.optimizer.yi))
        params = self.optimizer.Xi[np.argmin(self.optimizer.yi)]
        best_param_desr = "Best parameters:"
        for i, dim in enumerate(self.space):
            best_param_desr = best_param_desr + " " + dim.name + "=%s" % params[i]
        return best_param_desr



    def train_eval_best_model(self, data_sample_random_seeds=None):
        if len(self.best_model_train_eval) == self.num_samples_best_eval:
            print("Loaded best model eval for model: {model}, data set: {data_set}".format(model=FLAGS.model, data_set=FLAGS.data_set))
            return

        print("Running best model eval for model: {model}, data set: {data_set}".format(model=FLAGS.model,
                                                                                       data_set=FLAGS.data_set))

        if FLAGS.plot:
            progress = FloatProgress(min=0, max=1)
            display(progress)
        else:
            printProgressBar(0, self.num_samples_best_eval, prefix='Progress best model eval {model}/{data_set}:'.
                             format(model=FLAGS.model, data_set=FLAGS.data_set), suffix='Complete', length=50)
        futures = []
        done = 0.0

        with SameProcessExecutor() if self.num_workers <= 0 else concurrent.futures.ProcessPoolExecutor(
                self.num_workers) as executor:
            for i in range(self.num_samples_best_eval):
                inserted = False
                while not inserted:
                    if len(futures) < self.num_workers:
                        objective_fun = self.train_eval_model.create_train_eval(i)
                        params = self.best_model_params.copy()
                        params["tensorboard_folder"] = "tensorboard_best"
                        params["sample_cross_validation"] = True
                        params["data_sample_random_seed"] = generate_seed() if data_sample_random_seeds is None else data_sample_random_seeds[i]
                        params["tf_random_seed"] = generate_seed()
                        futures.append(
                            WorkItem(i, None, params, executor.submit(objective_fun, args=None, **params)))
                        inserted = True

                    for wi in list(futures):
                        try:
                            model_dir, train_eval, validation_eval, test_eval = wi.future.result(0)
                            self.best_task_eval_finished(futures, wi, model_dir, train_eval, validation_eval,
                                                         test_eval)
                            done += 1
                            if FLAGS.plot:
                                progress.value = done / self.num_samples_best_eval
                            else:
                                printProgressBar(done, self.num_samples_best_eval,
                                                 prefix='Progress best model eval {model}/{data_set}:'.
                                                 format(model=FLAGS.model, data_set=FLAGS.data_set), suffix='Complete',
                                                 length=50)
                        except concurrent.futures.TimeoutError:
                            pass

                    if len(futures) == self.num_workers:
                        time.sleep(5)

        for wi in list(futures):
            model_dir, train_eval, validation_eval, test_eval = wi.future.result()
            self.best_task_eval_finished(futures, wi, model_dir, train_eval, validation_eval,
                                         test_eval)
            done += 1
            if FLAGS.plot:
                progress.value = done / self.num_samples_best_eval
            else:
                printProgressBar(done, done,
                                 prefix='Progress best model eval {model}/{data_set}:'.
                                 format(model=FLAGS.model, data_set=FLAGS.data_set), suffix='Complete',
                                 length=50)

        predict_train, predict_valid, predict_test = invoke_in_process_pool(self.num_workers,
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.train_x,
                                                                                     self.train_eval_model.data_loader.train_y),
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.validation_x,
                                                                                     self.train_eval_model.data_loader.validation_y),
                                                                            Callable(predict_estimator, self.best_model,
                                                                                     self.train_eval_model.data_loader.test_x,
                                                                                     self.train_eval_model.data_loader.test_y)
                                                                            )

        self.best_model_train_ll = predict_train["log_likelihood"]
        self.best_model_valid_ll = predict_valid["log_likelihood"]
        self.best_model_test_ll = predict_test["log_likelihood"]

        self.save()

