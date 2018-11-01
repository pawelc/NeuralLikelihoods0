import datetime
import json

from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs

from data import DataLoader
import tensorflow as tf
import os

from experiment.exporters import BestResultExporter
from flags import FLAGS
from models.utils import get_train_inputs, get_eval_inputs, eval_estimator, add_debug_hooks
from tensorflow.python.platform import tf_logging as logging

from utils import create_session_config


class CheckNanTensorHook(session_run_hook.SessionRunHook):
    """NaN Loss monitor.
    """

    def __init__(self):
        pass

    def begin(self):
        self._check = tf.add_check_numerics_ops()

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._check)

    def after_run(self, run_context, run_values):
        logging.info(run_values.results)

class TrainEvalModel:

    def __init__(self, id, model_fn, data_loader: DataLoader, model_name, name_fields):
        self.id = id
        self.model_fn = model_fn
        self.data_loader = data_loader
        self.model_name = model_name
        self.name_fields = name_fields

    def name(self, id, args):
        name = 'id_{id}'.format(id=id)
        for field in self.name_fields:
            if field in args:
                val = args[field]
                if type(val) is list:
                    name = name + '_' + field + "_" + ("_".join([str(el) for el in val]))
                elif callable(val):
                    name = name + '_' + field + "_" + val.__name__
                else:
                    name = name + '_' + field + '_' + str(val)

        return name + '_ts_' + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    def create_estimator(self, model_dir, session_config, tf_random_seed, kargs,
                         save_checkpoints_steps=FLAGS.save_checkpoints_steps):
        run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                            save_summary_steps=FLAGS.save_summary_steps,
                                            save_checkpoints_steps=save_checkpoints_steps,
                                            save_checkpoints_secs=None,
                                            log_step_count_steps=1000, keep_checkpoint_max=3,
                                            session_config=session_config,
                                            tf_random_seed=tf_random_seed, )
        return tf.estimator.Estimator(config=run_config,
                                      model_fn=self.model_fn,
                                      params=kargs)

    def __call__(self, *varg, **kargs):
        tf.logging.set_verbosity(tf.logging.WARN)
        print("train_eval executing with params: %s" % json.dumps(kargs))
        try:
            tensorboard_folder = kargs["tensorboard_folder"]
            tf_random_seed = kargs["tf_random_seed"]
            batch_size = kargs["batch_size"]

            sample_cross_validation = True if ("sample_cross_validation" in kargs) and kargs[
                "sample_cross_validation"] else False

            if sample_cross_validation:
                data_sample_random_seed = kargs["data_sample_random_seed"]
                self.data_loader.sample_cross_validation(data_sample_random_seed)

            model_dir = tensorboard_folder + '/' + self.model_name + '/' + self.name(self.id, kargs)

            estimator = self.create_estimator(model_dir, create_session_config(), tf_random_seed, kargs)

            os.makedirs(estimator.eval_dir())

            early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
                estimator,
                metric_name='loss',
                max_steps_without_decrease=FLAGS.max_steps_without_decrease,
                run_every_steps=100,
                run_every_secs=None)

            hooks = [early_stopping_hook]
            add_debug_hooks(hooks)

            if FLAGS.check_nans:
                hooks.append(CheckNanTensorHook())
            train_spec = tf.estimator.TrainSpec(input_fn=lambda: get_train_inputs(self.data_loader, batch_size=batch_size), max_steps=None,
                                                hooks=hooks)

            hooks = []
            add_debug_hooks(hooks)

            exporter = BestResultExporter()
            eval_spec = tf.estimator.EvalSpec(
                input_fn=lambda: get_eval_inputs(self.data_loader, batch_size=batch_size), steps=None, start_delay_secs=0,
                throttle_secs=FLAGS.eval_throttle_secs, hooks=hooks, exporters=exporter)

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

            best_estimator = self.create_estimator(exporter.export_path, create_session_config(), tf_random_seed, kargs,
                                                   save_checkpoints_steps=None)

            train_eval = eval_estimator(best_estimator, self.data_loader.train_x, self.data_loader.train_y, batch_size=batch_size)
            validation_eval = eval_estimator(best_estimator, self.data_loader.validation_x,
                                             self.data_loader.validation_y, batch_size=batch_size)
            test_eval = eval_estimator(best_estimator, self.data_loader.test_x, self.data_loader.test_y, batch_size=batch_size)
            return model_dir, train_eval, validation_eval, test_eval
        except:
            print("ERROR train_and_evaluate with params: {params}, id: {id}, data seed: {data_seed}, "
                  "tf seed: {tf_random_seed}, model: {model}, data set: {data_set}".format(params=json.dumps(kargs),
                                                                                           id=self.id,
                                                                                           data_seed=str(
                                                                                               self.data_loader.random_seed),
                                                                                           tf_random_seed=kargs[
                                                                                               "tf_random_seed"],
                                                                                           model=FLAGS.model,
                                                                                           data_set=FLAGS.data_set))
            import logging as py_log
            py_log.exception('Got exception during training')
            return None, None, None, None


class TrainEvalModelFactory:

    def __init__(self, model_fn, data_loader: DataLoader, model_name, name_fields):
        self.model_fn = model_fn
        self.data_loader = data_loader
        self.model_name = model_name
        self.name_fields = name_fields

    def create_train_eval(self, id):
        return TrainEvalModel(id, self.model_fn, self.data_loader, self.model_name, self.name_fields)
