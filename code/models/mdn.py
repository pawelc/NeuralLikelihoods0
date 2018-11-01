import tensorflow as tf
import tensorflow_probability as tp
from models.utils import log_likelihood_from_cdfs_transforms, constrain_cdf, metric_loglikelihood, train_op, \
    print_tensor, add_all_summary_stats, extract_xy


def get_mixtures(output, num_marginals):
    marginal_params = tf.split(output, num_or_size_splits=num_marginals, axis=1)
    mixtures = []

    for marginal_id in range(num_marginals):
        with tf.variable_scope("mixture_%d"%marginal_id):
            out_logits, out_sigma, out_mu = tf.split(marginal_params[marginal_id], num_or_size_splits=3, axis=1)
            out_sigma = print_tensor(add_all_summary_stats(tf.maximum(1e-24,tf.square(out_sigma), name="sigma")))

            mix = tp.distributions.MixtureSameFamily(
                mixture_distribution=tp.distributions.Categorical(logits=out_logits),
                components_distribution=tp.distributions.Normal(loc=out_mu, scale=out_sigma))

            mixtures.append(mix)
    return mixtures

def get_loss(log_likelihood):
    return tf.negative(tf.reduce_mean(log_likelihood), name="loss")

def generate_ensemble(mix):
    return tf.reshape(mix.sample(1, name="sample"), [-1,1])

def compute_log_prob_mixture(mix, y, i):
    with tf.variable_scope("log_pdf_%d"%i):
        return tf.reshape(mix.log_prob(tf.reshape(y, [-1])),[-1,1])

def log_likelihood(cov_type, mixtures, y, lls,x, params):
    cdfs = []
    for i, mix in enumerate(mixtures):
        # this has to be 1-D so mixture computes parametrization per data point
        y_component = tf.reshape(tf.slice(y, [0, i], size=[-1, 1]), [-1], name="y_component%d"%i)
        cdfs.append(constrain_cdf(tf.reshape(mix.cdf(y_component, name="component_%d_cdf"%i),[-1,1])))

    return log_likelihood_from_cdfs_transforms(cov_type, cdfs, lls, x, params)

def compute_lls(mixtures, y):
    return [compute_log_prob_mixture(mix, tf.slice(y, [0, i], size=[-1, 1]), i) for i, mix in enumerate(mixtures)]

def mdn_model(features, labels, mode, params):
    k_mix = params["k_mix"]
    num_layers = params["num_layers"]
    hidden_units = params['hidden_units']
    learning_rate = params["learning_rate"]
    cov_type = params["cov_type"]

    x_size, y_size, x, y = extract_xy(features, params)
    layer = x

    n_out = k_mix * 3 * y_size  # pi, mu, stdev

    if x is not None:
        for i in range(num_layers):
            layer = tf.layers.dense(layer, units=hidden_units, activation=tf.nn.tanh, name="h_layer_%d" % i)

        Wo = tf.get_variable("Wo", shape=[params['hidden_units'], n_out], dtype=tf.float32)
        bo = tf.get_variable("bo", shape=[1, n_out], dtype=tf.float32)

        output = tf.matmul(layer, Wo) + bo
    else:
        output = tf.get_variable("output", shape=(1, n_out))

    mixtures = get_mixtures(output, y_size)

    if mode == tf.estimator.ModeKeys.PREDICT:
        if 'y' in "".join(features.keys()):
            marginal_lls = compute_lls(mixtures, y)
            predictions = {'pdf%d'%i: tf.exp(ll) for i,ll in enumerate(marginal_lls)}
            predictions["log_likelihood"] = log_likelihood(cov_type, mixtures, y, marginal_lls,x, params)
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            predictions = {'samples%d'%i:generate_ensemble(mix) for i, mix in enumerate(mixtures)}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    marginal_lls = compute_lls(mixtures, y)
    ll = log_likelihood(cov_type, mixtures, y, marginal_lls, x, params)
    loss = print_tensor(get_loss(ll))

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {"log_likelihood": metric_loglikelihood(ll)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op(loss, learning_rate))
