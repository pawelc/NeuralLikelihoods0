from models.nn_pdf_common import transform_x, create_cdf_layer_partial_monotonic_MLP, create_pdf_layer_mv, \
    create_partially_monotone_dense_layer, create_monotone_dense_layer
from models.utils import extract_xy, metric_loglikelihood, train_op, constrain_cdf, print_tensor
import tensorflow as tf
import numpy as np


def cdf_transform(x, y, params, positive_transform):
    x_transform = transform_x(params, x)

    y1 = tf.slice(y, [0, 0], [-1, 1])

    y2 = tf.slice(y, [0, 1], [-1, 1])

    xy1 = tf.concat([x_transform, y1], axis=1)
    xy2 = tf.concat([x_transform, y2], axis=1)

    activation1 = tf.nn.sigmoid
    activation2 = tf.nn.softplus

    arch2 = params['arch2']
    arch3 = params['arch3']
    #     relu_bias_initializer = float(params["relu_bias_initializer"])

    with tf.variable_scope("cdf_transform"):

        with tf.variable_scope("xy1_partially_monotone"):
            xy1 = create_partially_monotone_dense_layer(xy1, arch2[0], x_transform.shape[-1].value,
                                                        1, activation=activation1,
                                                        positive_transform=positive_transform)

        for i, units in enumerate(arch2[1:-1]):
            with tf.variable_scope("xy1_l_%d" % i):
                xy1 = create_monotone_dense_layer(xy1, units, activation=activation1,
                                                  positive_transform=positive_transform)

        with tf.variable_scope("xy1_l_%d" % len(arch2)):
            xy1_prod = create_monotone_dense_layer(xy1, arch2[-1], activation=activation1,
                                                   positive_transform=positive_transform)

        #         with tf.variable_scope("xy1_add_l_%d"%len(arch2)):
        #             xy1_add = create_monotone_dense_layer(xy1, arch2[-1], activation=activation1, positive_transform=positive_transform)

        with tf.variable_scope("xy2_partially_monotone"):
            xy2 = create_partially_monotone_dense_layer(xy2, arch2[0], x_transform.shape[-1].value,
                                                        1, activation=activation1,
                                                        positive_transform=positive_transform)

        for i, units in enumerate(arch2[1:-1]):
            with tf.variable_scope("xy2_l_%d" % i):
                xy2 = create_monotone_dense_layer(xy2, units, activation=activation1,
                                                  positive_transform=positive_transform)

        with tf.variable_scope("xy2_l_%d" % len(arch2)):
            xy2_prod = create_monotone_dense_layer(xy2, arch2[-1], activation=activation1,
                                                   positive_transform=positive_transform)

        #         with tf.variable_scope("xy2_add_l_%d"%len(arch2)):
        #             xy2_add = create_monotone_dense_layer(xy2, arch2[-1], activation=activation1, positive_transform=positive_transform)

        with tf.variable_scope("xy_prod"):
            xy_prefinal = tf.multiply(xy1_prod, xy2_prod)
            #             tf.concat([xy_prefinal, xy1_add,xy2_add], axis=1)
            xy_prefinal = create_monotone_dense_layer(xy_prefinal, arch3[0], activation=activation2,
                                                      positive_transform=positive_transform)

        for i, units in enumerate(arch3[1:]):
            with tf.variable_scope("xy_prefinal_l_%d" % i):
                xy_prefinal = create_monotone_dense_layer(xy_prefinal, units, activation=activation2,
                                                          positive_transform=positive_transform)

        return create_monotone_dense_layer(xy_prefinal, 1, activation=activation2,
                                           positive_transform=positive_transform)

def cdf_transforms(x, y, params, positive_transform, mode):
    with tf.variable_scope("extreme_vals",reuse=tf.AUTO_REUSE):
        y_max_values_vars = tf.get_variable("y_max_values_var", trainable=False, dtype=tf.float32,
                                            shape=(1, y.shape[-1].value), initializer=tf.constant_initializer(np.nan))

        if mode == tf.estimator.ModeKeys.TRAIN:
            y_max_values = tf.reduce_max(y, axis=0, keepdims=True)
            y_max_values_vars = tf.cond(tf.reduce_all(tf.is_nan(y_max_values_vars)), \
                  lambda: tf.assign(y_max_values_vars, y_max_values), \
                  lambda: tf.assign(y_max_values_vars, tf.reduce_max(tf.concat([y_max_values, y_max_values_vars], axis=0),axis=0, keepdims=True)))

    with tf.variable_scope("cdf", reuse=tf.AUTO_REUSE):
        non_normalized_cdf = cdf_transform(x, y, params, positive_transform)
        cdf_normalization = cdf_transform(x, tf.tile(y_max_values_vars, [tf.shape(x)[0], 1]),
                                                       params, positive_transform)
    return non_normalized_cdf, cdf_normalization


def nn_pdf_m_model(features, labels, mode, params):
    positive_transform = params['positive_transform']
    learning_rate = params["learning_rate"]

    x_size, y_size, x, y = extract_xy(features, params)

    y = print_tensor(y, name="y")

    if y is None or y.shape[-1].value != 2:
        raise NotImplementedError

    y_components = [tf.slice(y, [0,i],[-1,1]) for i in range(y_size)]
    y_components_combined = tf.concat(y_components, axis=1)

    non_normalized_cdf, cdf_normalization = cdf_transforms(x, y_components_combined, params, positive_transform, mode)

    non_normalized_cdf = tf.check_numerics(non_normalized_cdf, message="non_normalized_cdf: ")
    cdf_normalization = tf.check_numerics(cdf_normalization, message="cdf_normalization")

    grads = tf.maximum(create_pdf_layer_mv(non_normalized_cdf, y_components), 1e-24)

    log_likelihood = tf.log(grads) - tf.log(tf.maximum(cdf_normalization, 1e-24))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"log_likelihood": log_likelihood, "cdf":tf.div(non_normalized_cdf,cdf_normalization)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = print_tensor(tf.negative(tf.reduce_mean(log_likelihood), name="loss"))

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {"log_likelihood": metric_loglikelihood(log_likelihood)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op(loss, learning_rate))






