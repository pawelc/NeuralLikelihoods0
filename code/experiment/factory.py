from skopt.space import Integer, Real, Categorical

import experiment
from data.registry import create_data_loader
from experiment.experiment import Runner
from experiment.train_eval import TrainEvalModelFactory
from flags import FLAGS
from models import nn_pdf_model, mdn_model, nn_pdf_ar_model
from models.nn_pdf_m import nn_pdf_m_model
from models.rnade import rnade_model
from models.rnade_deep import rnade_deep_model
import tensorflow as tf


def create_run_experiment():
    data_loader = create_data_loader()
    exp_factory_clazz = getattr(experiment.factory, FLAGS.model)
    exp_factory = exp_factory_clazz()
    exp = None
    if exp_factory.is_compatible_with(data_loader):
        exp = exp_factory.create(data_loader, FLAGS.num_workers, FLAGS.num_samples, FLAGS.num_samples_best_eval)
        exp.load_best_model_or_run()
    return exp

def create_load_experiment():
    data_loader = create_data_loader()
    exp_factory_clazz = getattr(experiment.factory, FLAGS.model)
    exp_factory = exp_factory_clazz()
    exp = None
    if exp_factory.is_compatible_with(data_loader):
        exp = exp_factory.create(data_loader, FLAGS.num_workers, FLAGS.num_samples, FLAGS.num_samples_best_eval)
        exp.load_best_model()
    return exp


class nn_pdf_const_cov:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_nn_pdf_experiment("const_cov", data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True


class nn_pdf_param_cov:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_nn_pdf_experiment("param_cov", data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True if data_loader.train_y.shape[1] > 1 else False

class nn_pdf_ar:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_nn_pdf_ar_experiment(data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True if data_loader.train_y.shape[1] > 1 else False

class nn_pdf_m:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_nn_pdf_m_experiment(data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True if data_loader.train_y.shape[1] == 2 else False

class mdn_const_cov:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_mdn_experiment("const_cov", data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True


class mdn_param_cov:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_mdn_experiment("param_cov", data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True if data_loader.train_y.shape[1] > 1 else False


class rnade_laplace:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_rnade_experiment("laplace", data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True

class rnade_deep_normal:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_rnade_deep_experiment("normal",data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True

class rnade_deep_laplace:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_rnade_deep_experiment("laplace",data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True


class rnade_normal:

    def create(self, data_loader, num_workers, num_samples, num_samples_best_eval):
        return create_rnade_experiment("normal", data_loader, num_workers, num_samples, num_samples_best_eval)

    def is_compatible_with(self, data_loader):
        return True


def nn_pdf_train_eval_model_factory(data_loader, name="debug"):
    return TrainEvalModelFactory(
        nn_pdf_model,
        data_loader,
        "nn_pdf_" + name,
        ['arch1', 'arch2',"arch_cov" ,"batch_size", "learning_rate","activation"])


def create_nn_pdf_experiment(cov_type, data_loader, num_workers, num_samples, num_samples_best_eval):
    def params_converter(args):
        if data_loader.train_x.shape[1] > 0:
            args['arch1'] = [args['layer_size1']] * args['num_layers1']

        args['arch2'] = [args['layer_size2']] * args['num_layers2']
        args['positive_transform'] = "square"
        args['batch_size'] = 200
        args['cov_type'] = cov_type
        # args['activation'] = tf.nn.relu

        if cov_type == "param_cov":
            args['arch_cov'] = [args['layer_size_cov']] * args['num_layers_cov']

        return args

    opt_space = [
        Integer(1, 5, name='num_layers2'),
        Integer(10, 200, name="layer_size2"),
        Real(10 ** -4, 10 ** -2, "log-uniform", name='learning_rate'),
        # Categorical(["relu","tanh","leaky_relu"], name="activation")
        # Integer(100, 500, name="batch_size"),
    ]

    if data_loader.train_x.shape[1] > 0:
        opt_space.extend([Integer(0, 3, name='num_layers1'), Integer(10, 200, name="layer_size1")])

    if cov_type == "param_cov":
        opt_space.extend([Integer(1, 3, name='num_layers_cov'), Integer(10, 200, name="layer_size_cov")])

    opt = Runner(num_workers=num_workers,
                 num_samples=num_samples,
                 num_samples_best_eval=num_samples_best_eval,
                 space=opt_space,
                 params_converter=params_converter,
                 train_eval_model=nn_pdf_train_eval_model_factory(data_loader, cov_type))

    opt.load_best_model_or_run()
    return opt

def nn_pdf_ar_train_eval_model_factory(data_loader):
    return TrainEvalModelFactory(
        nn_pdf_ar_model,
        data_loader,
        "nn_pdf_ar",
        ['arch1', 'arch2', "batch_size", "learning_rate"])

def nn_pdf_m_train_eval_model_factory(data_loader):
    return TrainEvalModelFactory(
        nn_pdf_m_model,
        data_loader,
        "nn_pdf_m",
        ['arch1','arch2' ,'arch3', "batch_size", "learning_rate","relu_bias_initializer"])

def create_nn_pdf_m_experiment(data_loader, num_workers, num_samples, num_samples_best_eval):
    def params_converter(args):
        args['arch1'] = [args['layer_size1']] * args['num_layers1']
        args['arch2'] = [args['layer_size2']] * args['num_layers2']
        args['arch3'] = [args['layer_size3']] * args['num_layers3']
        args['positive_transform'] = "square"
        args['batch_size'] = 200
        return args

    opt_space = [
        Real(10 ** -4, 10 ** -3, "log-uniform", name='learning_rate'),
        Integer(1, 3, name='num_layers1'),
        Integer(10, 200, name="layer_size1"),

        Integer(1, 3, name='num_layers2'),
        Integer(10, 200, name="layer_size2"),

        Integer(1, 3, name='num_layers3'),
        Integer(10, 200, name="layer_size3"),
    ]

    opt = Runner(num_workers=num_workers,
                 num_samples=num_samples,
                 num_samples_best_eval=num_samples_best_eval,
                 space=opt_space,
                 params_converter=params_converter,
                 train_eval_model=nn_pdf_m_train_eval_model_factory(data_loader))

    opt.load_best_model_or_run()
    return opt

def create_nn_pdf_ar_experiment(data_loader, num_workers, num_samples, num_samples_best_eval):
    def params_converter(args):
        args['arch1'] = [args['layer_size1']] * args['num_layers1']
        args['arch2'] = [args['layer_size2']] * args['num_layers2']
        args['positive_transform'] = "square"
        args['batch_size'] = 200

        return args

    opt_space = [
        Integer(1, 5, name='num_layers2'),
        Integer(10, 200, name="layer_size2"),
        Real(10 ** -4, 10 ** -2, "log-uniform", name='learning_rate'),
        Integer(0, 3, name='num_layers1'),
        Integer(10, 200, name="layer_size1")
        # Integer(100, 500, name="batch_size"),
    ]

    opt = Runner(num_workers=num_workers,
                 num_samples=num_samples,
                 num_samples_best_eval=num_samples_best_eval,
                 space=opt_space,
                 params_converter=params_converter,
                 train_eval_model=nn_pdf_ar_train_eval_model_factory(data_loader))

    opt.load_best_model_or_run()
    return opt


def mdn_train_eval_model_factory(data_loader, cov_type="debug"):
    return experiment.train_eval.TrainEvalModelFactory(mdn_model,
                                                       data_loader,
                                                       "mdn_" + cov_type,
                                                       ["num_layers", "hidden_units", "k_mix", "batch_size",
                                                        "learning_rate"])


def create_mdn_experiment(cov_type, data_loader, num_workers, num_samples, num_samples_best_eval):
    def params_converter(args):
        # args['batch_size'] = data_loader.train_x.shape[0]
        args['batch_size'] = 200
        args['cov_type'] = cov_type

        if cov_type == "param_cov":
            args['arch_cov'] = [args['layer_size_cov']] * args['num_layers_cov']
        return args

    opt_space = [Integer(1, 6, name='num_layers'),
                                              Integer(20, 200, name="hidden_units"),
                                              Integer(1, 100, name="k_mix"),
                                              Real(10 ** -4, 10 ** -2, "log-uniform", name='learning_rate')
                                              ]

    if cov_type == "param_cov":
        opt_space.extend([Integer(1, 3, name='num_layers_cov'), Integer(10, 200, name="layer_size_cov")])

    opt = experiment.experiment.Runner(num_workers=num_workers,
                                       num_samples=num_samples,
                                       num_samples_best_eval=num_samples_best_eval,
                                       space=opt_space,
                                       params_converter=params_converter,
                                       train_eval_model=mdn_train_eval_model_factory(data_loader, cov_type))

    opt.load_best_model_or_run()
    return opt


def create_rnade_experiment(components_distribution, data_loader, num_workers, num_samples, num_samples_best_eval):
    def params_converter(args):
        args['batch_size'] = 200
        args['components_distribution'] = components_distribution
        return args

    opt = experiment.experiment.Runner(num_workers=num_workers,
                                       num_samples=num_samples,
                                       num_samples_best_eval=num_samples_best_eval,
                                       space=[Integer(20, 200, name="hidden_units"),
                                              Integer(1, 100, name="k_mix"),
                                              Real(10 ** -4, 10 ** -2, "log-uniform", name='learning_rate'),
                                              ],
                                       params_converter=params_converter,
                                       train_eval_model=rnade_train_eval_model_factory(data_loader,components_distribution))

    opt.load_best_model_or_run()
    return opt

def create_rnade_deep_experiment(components_distribution, data_loader, num_workers, num_samples, num_samples_best_eval):
    def params_converter(args):
        args['batch_size'] = 200
        args['arch'] = [args['layer_size']] * args['num_layers']
        args['components_distribution'] = components_distribution
        return args

    opt = experiment.experiment.Runner(num_workers=num_workers,
                                       num_samples=num_samples,
                                       num_samples_best_eval=num_samples_best_eval,
                                       space=[
                                           Integer(1, 5, name='num_layers'),
                                           Integer(10, 200, name="layer_size"),
                                              Integer(1, 100, name="k_mix"),
                                              Integer(1,5, name="num_ensambles"),
                                              Real(10 ** -4, 10 ** -2, "log-uniform", name='learning_rate'),
                                              ],
                                       params_converter=params_converter,
                                       train_eval_model=rnade_deep_train_eval_model_factory(data_loader,components_distribution))

    opt.load_best_model_or_run()
    return opt


def rnade_train_eval_model_factory(data_loader, name="debug"):
    return experiment.train_eval.TrainEvalModelFactory(
        rnade_model,
        data_loader,
        "rnade_" + name,
        ["hidden_units", "k_mix", "batch_size", 'learning_rate'])

def rnade_deep_train_eval_model_factory(data_loader, name="debug"):
    return experiment.train_eval.TrainEvalModelFactory(
        rnade_deep_model,
        data_loader,
        "rnade_deep" + name,
        ["arch", "k_mix", "batch_size", 'learning_rate','num_ensambles'])
