import os
import pickle
from tensorflow.python import debug as tf_debug

import numpy as np
import tensorflow as tf
import tensorflow_probability as tp
import re

import models
from flags import FLAGS
from utils import create_session_config

def get_activation(params):
    activation = params["activation"] if "activation" in params else "tanh"
    if activation == "relu":
        return tf.nn.relu
    elif activation == "tanh":
        return tf.nn.tanh
    elif activation == "leaky_relu":
        return tf.nn.leaky_relu
    else:
        return tf.nn.tanh

def match_list_of_regexp_to_string(list_regexp, text):
    return np.any([re.search(reg_exp, text) for reg_exp in list_regexp])

def print_tensor(tensor, name=None):
    if FLAGS.print_tensors:
        if name is None:
            name =  tensor.name
        if FLAGS.print_tensors_filter=="" or match_list_of_regexp_to_string(FLAGS.print_tensors_filter.split(","), name):
            # name = tf.get_variable_scope().name + "/" + name
            return tf.Print(tensor, [tensor], message="%s: " % name, summarize=FLAGS.print_tensors_summarize)

    return tensor

def print_tensors(tensor, tensors, name=None):
    if FLAGS.print_tensors:
        if name is None:
            name = tensor.name
        # name = tf.get_variable_scope().name + "/" + name
        return tf.Print(tensor, tensors, message="%s: " % name, summarize=1000000)
    else:
        return tensor


def unpack_data(name, data):
    if data is None:
        return None
    return {name + "%d" % i: data[:, i] for i in range(data.shape[1])}


def get_train_inputs(data_loader, batch_size=None):
    x = data_loader.train_x
    y = data_loader.train_y
    max_num_epochs = None if FLAGS.max_num_epochs == -1 else FLAGS.max_num_epochs


    dataset = tf.data.Dataset.from_tensor_slices(({'x': x, 'y': y}))
    dataset = dataset.shuffle(
        buffer_size=1000, reshuffle_each_iteration=True
    ).repeat(count=max_num_epochs).batch(x.shape[0] if batch_size is None else batch_size).prefetch(1)

    return dataset


def get_eval_inputs(data_loader, batch_size=None):
    x = data_loader.validation_x
    y = data_loader.validation_y

    dataset = tf.data.Dataset.from_tensor_slices(({'x': x, 'y': y}))
    dataset = dataset.batch(x.shape[0] if batch_size is None else batch_size).prefetch(1)
    return dataset


def get_inputs(x, y, batch_size=None):
    data = {}
    data_size = None
    if x is not None:
        data['x'] = x
        data_size = x.shape[0]
    if y is not None:
        data['y']=y
        data_size = y.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((data))
    dataset = dataset.batch(data_size if batch_size is None else batch_size).prefetch(1)
    return dataset

def best_model_file(name):
    return 'best_model_%s.pkl' % name

def save_best_model_exp(name, opt):
    with open(best_model_file(name), 'wb') as f:
        data = {}
        data["res"] = opt.res
        data["best_model_loss"] = opt.best_model_loss
        data["best_model_params"] = opt.best_model_params
        data["best_model_dir"] = opt.best_model_dir
        data["best_model"] = opt.best_model
        data["best_model_train_eval"] = opt.best_model_train_eval
        data["best_model_validation_eval"] = opt.best_model_validation_eval
        data["best_model_test_eval"] = opt.best_model_test_eval
        data["best_model_train_ll"] = opt.best_model_train_ll
        data["best_model_valid_ll"] = opt.best_model_valid_ll
        data["best_model_test_ll"] = opt.best_model_test_ll

        data["best_model"] = opt.best_model
        data["optimizer"] = opt.optimizer

        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def add_debug_hooks(hooks):
    if FLAGS.debug_tb:
        debug_hook = tf_debug.TensorBoardDebugHook("pawel-workstation:8080")
        hooks.append(debug_hook)
    elif FLAGS.debug_cli:
        debug_hook = tf_debug.LocalCLIDebugHook()
        debug_hook.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        hooks.append(debug_hook)

def eval_estimator(estimator, x, y, batch_size=None):
    hooks = []
    add_debug_hooks(hooks)
    return estimator.evaluate(input_fn=lambda: get_inputs(x, y, batch_size=batch_size), steps=1, hooks=hooks)


def predict_estimator(estimator, x, y, batch_size=None):
    hooks=[]
    add_debug_hooks(hooks)
    return next(estimator.predict(input_fn=lambda: get_inputs(x, y, batch_size=batch_size), yield_single_examples=False, hooks=hooks))


def retrieve_vars(best_model_dir):
    vars = []
    tf.reset_default_graph()

    checkpoint = tf.train.get_checkpoint_state(best_model_dir)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # just to check all variables values

        for var in tf.all_variables():
            shape = var.shape.as_list()
            if "Adam" not in var.name and "global_step" not in var.name and "_power" not in var.name and "early_stopping" not in var.name:
                vars.append([var.name, sess.run(var)])

    return vars

def show_exps(*experiments):
    for experiment in experiments:
        print(
            "Experiment {model_name} with results {results}".format(model_name=experiment["name"], results=experiment))


def tf_cov(x):
    with tf.variable_scope("covariance"):
        mean_x = print_tensor(tf.reduce_mean(x, axis=0, keepdims=True, name="mean"))
        mx = tf.matmul(tf.transpose(mean_x), mean_x, name="mx")
        vx = tf.divide(tf.matmul(tf.transpose(x), x), tf.cast(tf.shape(x)[0], tf.float32), name="vx")
        cov_xx = tf.subtract(vx, mx, name="cov")
        return cov_xx


def corr(cov):
    diag = print_tensor(tf.diag(tf.diag_part(cov)), "diag")
    D = print_tensor(tf.sqrt(diag),"D")
    DInv = tf.matrix_inverse(D)
    corr = tf.matmul(tf.matmul(DInv, cov), DInv)
    corr = tf.matrix_set_diag(corr, [1.] * corr.shape[0].value, name="correlation")
    return corr


def assert_cov_positive_definite(cov):
    cov = print_tensor(cov, "cov")
    e, v = tf.self_adjoint_eig(cov)
    i = tf.constant(0)

    def adjust_cov(i, cov, eig_val):
        eig_val_max = tf.maximum(1e-6, tf.abs(eig_val))
        cov1 = cov + tf.eye(cov.shape[0].value) * tf.abs(eig_val_max)
        cov1_print = print_tensors(cov1, [i, eig_val, eig_val_max, cov, cov1],name="cov_adjust")
        e, v = tf.self_adjoint_eig(cov1_print)
        ePrint = print_tensors(e[0], [e[0]], name="smallest_eig")
        return tf.add(i, 1), cov1_print, ePrint

    def cond(i, cov, eig_val):
        return tf.less_equal(eig_val, 1e-6)

    i, cov, _ = tf.while_loop(cond, adjust_cov, [i, cov, e[0]])
    return cov

def covariance(x, dim, params):
    activation = get_activation(params)
    if x is not None:
        layer=x

        if "arch_cov" in params:
            for i, units in enumerate(params["arch_cov"]):
                layer = tf.layers.dense(layer, units=units, activation=activation, name="cov_layer_%d" % i)
                last_hidden_units_size=units

            layer_u = tf.slice(layer, [0,0], [-1,int(last_hidden_units_size/2)])
            layer_d = tf.slice(layer, [0,int(last_hidden_units_size/2)], [-1,-1])

            W_cov_u = tf.get_variable("W_cov_u", shape=[int(last_hidden_units_size/2), dim], dtype=tf.float32)
            b_cov_u = tf.get_variable("b_cov_u", shape=[1, dim], dtype=tf.float32)

            cov_u = tf.expand_dims(tf.matmul(layer_u, W_cov_u) + b_cov_u, 1)

            W_cov_d = tf.get_variable("W_cov_d", shape=[last_hidden_units_size-int(last_hidden_units_size/2), dim], dtype=tf.float32)
            b_cov_d = tf.get_variable("b_cov_d", shape=[1, dim], dtype=tf.float32)

            cov_d = tf.add(1e-5 ,tf.square(tf.matmul(layer_d, W_cov_d) + b_cov_d))
        else:
            for i in range(2):
                layer = tf.layers.dense(layer, units=20, activation=activation, name="cov_layer_%d" % i)

            W_cov_u = tf.get_variable("W_cov_u", shape=[20, dim], dtype=tf.float32)
            b_cov_u = tf.get_variable("b_cov_u", shape=[1, dim], dtype=tf.float32)

            cov_u = tf.expand_dims(tf.matmul(layer, W_cov_u) + b_cov_u, 1)

            W_cov_d = tf.get_variable("W_cov_d", shape=[20, dim], dtype=tf.float32)
            b_cov_d = tf.get_variable("b_cov_d", shape=[1, dim], dtype=tf.float32)

            cov_d = tf.add(1e-6, tf.square(tf.matmul(layer, W_cov_d) + b_cov_d))
    else:
        cov_u = tf.get_variable("cov_u", shape=(1, dim))
        cov_d = tf.add(1e-5 ,tf.square(tf.get_variable("cov_d_raw", shape=(1, dim)), name="cov_d"))

    diagonal = tf.matrix_diag(cov_d)

    return tf.add(tf.matmul(cov_u, cov_u, transpose_a=True), diagonal, name="covariance")

def correlation_matrix(cov):
    std_dev = tf.matrix_diag(tf.sqrt(tf.matrix_diag_part(cov)))
    std_dev_inv = tf.matrix_inverse(std_dev)

    corr = tf.matmul(tf.matmul(std_dev_inv, cov), std_dev_inv)

    return tf.matrix_set_diag(corr, tf.fill(tf.slice(tf.shape(corr), [0], [2]), 1.0), name="correlation")

def log_likelihood_from_cdfs_transforms(cov_type, cdfs, lls, x, params):
    ll_marginals_sum = tf.reshape(tf.reduce_sum(tf.concat(lls, axis=1), axis=1), [-1, 1])
    if len(lls) == 1:
        return ll_marginals_sum

    quantiles = []
    std_normal = tp.distributions.Normal(loc=0., scale=1.)
    for i, cdf in enumerate(cdfs):
        with tf.variable_scope("quantile_%d" % i):
            cdf = print_tensor(cdf, "cdf")
            quantiles.append(std_normal.quantile(cdf, name="quantile"))

    quantiles_combined = print_tensor(tf.concat(quantiles, axis=1, name="quantiles_combined"))

    if cov_type=="param_cov":
        cov = covariance(x, len(cdfs), params)
        cor = correlation_matrix(cov)
    elif cov_type=="const_cov":
        cov = tf_cov(quantiles_combined)
        cov = assert_cov_positive_definite(cov)
        cor = corr(cov)
        cor = tf.expand_dims(cor, axis=0)
    else:
        raise ValueError("Not recognized covariance type: %s"%cov_type)

    # c_pdf = tp.distributions.MultivariateNormalFullCovariance(loc=tf.constant([0.0] * len(cdfs)),
    #                                                           covariance_matrix=cor)
    # c_pdf_log_prob = tf.reshape(c_pdf.log_prob(tf.stop_gradient(quantiles_combined)), [-1, 1], name="copula_log_prob")

    c_pdf_log_prob = -tf.log(tf.sqrt(tf.matrix_determinant(cor)))
    inv_cor_min_eye = tf.matrix_inverse(cor) - tf.eye(len(cdfs))
    inv_cor_min_eye = tf.tile(inv_cor_min_eye,
                              [tf.div(tf.shape(quantiles_combined)[0], tf.shape(inv_cor_min_eye)[0]), 1, 1])
    c_pdf_log_exponent = tf.matmul(tf.expand_dims(quantiles_combined, axis=1), inv_cor_min_eye)
    c_pdf_log_exponent = - 0.5 * tf.matmul(c_pdf_log_exponent, tf.expand_dims(quantiles_combined, axis=-1))
    c_pdf_log_exponent = tf.reshape(c_pdf_log_exponent, [-1, 1])
    c_pdf_log_prob = tf.reshape(c_pdf_log_prob, [-1,1]) + c_pdf_log_exponent

    return tf.add(c_pdf_log_prob, ll_marginals_sum)


def constrain_cdf(cdf, name="cdf"):
    cdf = tf.minimum(1 - 1e-6, cdf)
    cdf = tf.maximum(1e-36, cdf, name=name)
    return cdf


def metric_loglikelihood(ll):
    return tf.metrics.mean(ll)


def generate_seed():
    random_data = os.urandom(4)
    return int.from_bytes(random_data, byteorder="big")


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # , end='\r'
    # Print New Line on Complete
    # if iteration == total:
    #     print()

def compute_mixture_of_gaussians(pi, mu, sigma, y):
    return tf.reduce_sum(tf.multiply(pi, tf.distributions.Normal(loc=mu, scale=sigma).prob(y)), axis=1, keepdims=True)

def compute_mixture_of_laplacians(pi, mu, sigma, y, name="mixture_of_laplacians"):
    return print_tensor(add_all_summary_stats(
        tf.reduce_sum(tf.multiply(pi, tf.distributions.Laplace(loc=mu, scale=sigma).prob(y)), axis=1, keepdims=True, name=name)))

def train_op(loss, learning_rate):
    if FLAGS.debug_grad:
        with tf.name_scope("train_op"):
            trainables = tf.trainable_variables()

            grads = tf.gradients(loss, trainables)

            # grads, _ = tf.clip_by_global_norm(grads, clip_norm=1)

            grads_printed = []
            for grad, var in zip(grads, trainables):
                print(var.name, var, grad.name, grad)
                grads_printed.append(print_tensor(grad, name="grad_%s"%var.name))

            opt = tf.train.GradientDescentOptimizer(learning_rate)

            train_op = opt.apply_gradients(zip(grads_printed, trainables), global_step=tf.train.get_global_step())
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)
        # optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 2.)
        gvs = optimizer.compute_gradients(loss)

        grads,vars = zip(*gvs)
        clipped_grads = []

        for grad, var in gvs:
            # clipped_grads.append(tf.clip_by_norm(grad, 5.))
            print_tensor(add_all_summary_stats(grad, "%s_grad"%var.name),name="%s_grad"%var.name)
            # print_tensor(add_all_summary_stats(clipped_grads[-1], "%s_clipped_grad" % var.name), name="%s_clipped_grad" % var.name)
        #     zip(clipped_grads, vars)

        train_op = optimizer.apply_gradients(gvs, global_step=tf.train.get_global_step())

    return train_op

def add_all_summary_stats(tensor, name=None):
    if FLAGS.summary:
        if name is None:
            name = tensor.name

        if FLAGS.summary_filter == "" or match_list_of_regexp_to_string(FLAGS.summary_filter.split(","),
                                                                              name):
            with tf.name_scope(("summaries_%s"%name).replace(":","_")):
                tf.summary.scalar("%s_min" % name, tf.reduce_min(tensor))
                tf.summary.scalar("%s_max" % name, tf.reduce_max(tensor))
                tf.summary.scalar("%s_norm" % name, tf.global_norm([tensor]))
                mean = tf.reduce_mean(tensor)
                tf.summary.scalar("%s_mean"% name, mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                tf.summary.scalar("%s_stddev"% name, stddev)
                tf.summary.scalar("%s_NaNs"% name, tf.reduce_sum(tf.to_float(tf.is_nan(tensor))))
                tf.summary.histogram("%s_hist"% name, tensor)
    return tensor

def load_model(model_dir, params):
    model_fn = getattr(models, "%s_model"%FLAGS.model)

    run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                        save_summary_steps=FLAGS.save_summary_steps, save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                        log_step_count_steps=1000, keep_checkpoint_max=3,
                                        session_config=create_session_config())
    return tf.estimator.Estimator(model_dir=model_dir, model_fn=model_fn, params=params, config=run_config)

def extract_xy(features, params):
    x_size = params["x_size"]
    y_size = params["y_size"]
    x,y=None,None
    if x_size > 0:
        x_columns = [tf.feature_column.numeric_column("x", shape=[x_size])]
        x = print_tensor(tf.feature_column.input_layer(features, x_columns))

    if y_size > 0:
        y_columns = [tf.feature_column.numeric_column("y", shape=[y_size])]
        y = tf.feature_column.input_layer(features, y_columns) if 'y' in features else None

    return x_size, y_size,x, y
