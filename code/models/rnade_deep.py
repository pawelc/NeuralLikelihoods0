from models.utils import extract_xy, metric_loglikelihood, train_op, constrain_cdf, print_tensor
import tensorflow as tf
import tensorflow_probability as tp
import numpy as np

def sample_idx(y_size):
    idx_sampler = tp.distributions.Categorical(probs=np.ones(y_size) / y_size, allow_nan_stats=False)
    idx = idx_sampler.sample()
    return idx

def sample_ordering(y_size):
    return tf.random_shuffle(tf.range(0, y_size))

def ll_for_random_ordering(x, idx, ordering, y, y_size, params, eval_only_first=False):
    with tf.variable_scope("rnade_deep", reuse=tf.AUTO_REUSE):
        arch = params['arch']
        k_mix = params['k_mix']
        components_distribution_param = params["components_distribution"]
        input_y = y
        output_y = y
        # masking input
        mask_idx = tf.slice(ordering, [idx], [-1])
        mask = tf.where(tf.reduce_any(tf.equal(tf.reshape(tf.range(0, y_size), [-1, 1]), mask_idx), axis=1),
                                       x=tf.zeros(y_size, dtype=tf.float32), y=tf.ones(y_size, dtype=tf.float32))
        mask_matrix = tf.diag(mask)
        input_y = tf.matmul(input_y, mask_matrix)

        input_y = tf.concat([input_y, tf.tile(tf.expand_dims(mask,0),[tf.shape(y)[0],1])], axis=1)
        if x is not None:
            layer = tf.concat([x, input_y], axis=1)
        else:
            layer = input_y

        for i,units in enumerate(arch):
            layer = tf.layers.dense(layer, units, activation=tf.nn.relu, name="layer_%d"%i)

        # param layer
        n_params_for_component = (1 + 1 + 1) * k_mix  # mean, scale and mixture
        param_layer = tf.layers.dense(layer, y_size * n_params_for_component)

        log_prob_sum = 0.0

        if eval_only_first:
            mask_idx = tf.slice(mask_idx, [0], [1])

        for comp_idx in range(y_size):
            with tf.variable_scope("comp_%d" % comp_idx):
                comp_start_idx = comp_idx * n_params_for_component
                mu = tf.slice(param_layer, [0, comp_start_idx], [-1, k_mix])

                z_scale = tf.slice(param_layer, [0, comp_start_idx + k_mix], [-1, k_mix])
                z_alpha = tf.slice(param_layer, [0, comp_start_idx + 2 * k_mix], [-1, k_mix])

                scale = tf.maximum(1e-24, tf.square(z_scale))

                if components_distribution_param == "normal":
                    components_distribution = tp.distributions.Normal(loc=mu, scale=scale)
                elif components_distribution_param == "laplace":
                    components_distribution = tp.distributions.Laplace(loc=mu, scale=scale)
                else:
                    raise ValueError(components_distribution_param)

                mix = tp.distributions.MixtureSameFamily(
                    mixture_distribution=tp.distributions.Categorical(logits=z_alpha, allow_nan_stats=False),
                    components_distribution=components_distribution, allow_nan_stats=False)

                log_prob_sum = tf.add(log_prob_sum, tf.cond(tf.reduce_any(tf.equal(comp_idx, mask_idx)), lambda: mix.log_prob(
                    tf.reshape(tf.slice(output_y, [0, comp_idx], [-1, 1]), [-1])       ), lambda: 0.0))
        return tf.reshape(log_prob_sum, [-1,1])


def compute_ensamble(params, x,y, y_size):
    num_ensambles = params["num_ensambles"]
    ensambles = []
    for e_idx in range(num_ensambles):
        ordering = sample_ordering(y_size)
        ensambles.append(tf.add_n(
            [ll_for_random_ordering(x, i, ordering, y, y_size, params, eval_only_first=True) for i in range(y_size)]))
    return tf.reduce_mean(tf.concat(ensambles, axis=1), axis=1)


def rnade_deep_model(features, labels, mode, params):
    x_size, y_size, x, y = extract_xy(features, params)
    learning_rate = params['learning_rate']

    if mode == tf.estimator.ModeKeys.PREDICT:
        ll = compute_ensamble(params, x,y, y_size)
        predictions = {"log_likelihood": ll, "pdf": tf.exp(ll)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    if mode == tf.estimator.ModeKeys.EVAL:
        ll = compute_ensamble(params, x, y, y_size)
        metrics = {"log_likelihood": metric_loglikelihood(ll)}
        return tf.estimator.EstimatorSpec(mode, loss=tf.negative(tf.reduce_mean(ll)), eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    idx = sample_idx(y_size)
    ordering = sample_ordering(y_size)
    ll = ll_for_random_ordering(x, idx, ordering, y, y_size, params)
    loss = tf.cast(y_size, dtype=tf.float32) / tf.cast(y_size - idx, tf.float32) * tf.negative(tf.reduce_mean(ll))

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op(loss, learning_rate))






