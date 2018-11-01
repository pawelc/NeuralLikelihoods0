import tensorflow as tf

from models.utils import constrain_cdf, add_all_summary_stats, print_tensor, get_activation


def create_weights(shape):
    w=tf.get_variable(name="weights", shape=shape, dtype=tf.float32)
    add_all_summary_stats(w)
    return w

def create_bias(units=None, initializer=None, clip_min = None, clip_max = None):
    with tf.name_scope('biases'):
        if initializer == None:
            shape = (units,)
        else:
            shape = None

        if clip_min is None:
            clip_min = -1000
        if clip_max is None:
            clip_max = 1000


        b = tf.clip_by_value(tf.get_variable(shape=shape, dtype=tf.float32, name="bias", initializer=initializer),
                             clip_value_min=clip_min, clip_value_max=clip_max)


        add_all_summary_stats(b)
    return b

def create_positive_weights(shape, positive_transform):
    w=tf.get_variable(name="weights-raw", shape=shape, dtype=tf.float32)
    # initializer=tf.initializers.truncated_normal(0.001, 0.1)
    add_all_summary_stats(w)

    if positive_transform == "exp":
        w_positive = tf.exp(w, name='weights-positive')
    elif positive_transform == "square":
        w_positive = tf.square(w, name='weights-positive')
    elif positive_transform == "softplus":
        w_positive = tf.nn.softplus(w, name='weights-positive')
    else:
        raise ValueError("wrong positive_transform: %s"%positive_transform)

    add_all_summary_stats(w_positive)
    return w_positive


def transform_x(params, x):
    x_transform = x
    activation = get_activation(params)
    if x is not None:
        for i, units in enumerate(params['arch1']):
            x_transform = print_tensor(tf.layers.dense(x_transform, units=units, activation=activation, name="x_transform_%d" % i), name="x_transform_%d" % i)
    return x_transform


def create_pdf_layer(cdf, y):
    with tf.name_scope("pdf_layer"):
        assert cdf.shape[0].value == y.shape[0].value
        assert y.shape[-1].value == 1
        assert cdf.shape[-1].value == 1
        # because it is list and cdf is only 1 element
        return tf.gradients(cdf, y)[0]

def create_pdf_layer_mv(cdf, y_components):
    gradients = cdf
    for i, y in enumerate(y_components):
        with tf.name_scope("pdf_layer_mv_%d"%i):
            gradients = create_pdf_layer(gradients, y)
    return gradients


def create_partially_monotone_dense_layer(input, units, num_non_mon_vars, num_mon_vars, activation, positive_transform,
                                          bias_initializer = None):
    if num_non_mon_vars>0:
        with tf.variable_scope("non_mon_weights"):
            w_non_mon = create_weights(shape=[num_non_mon_vars, units])
    with tf.variable_scope("mon_weights"):
        w_mon = create_positive_weights(shape=[num_mon_vars, units],positive_transform=positive_transform)

    if num_non_mon_vars > 0:
        w = tf.concat([w_non_mon,w_mon], axis=0)
    else:
        w = w_mon

    if bias_initializer is None:
        b = create_bias(units)
    else:
        b = create_bias(initializer=[bias_initializer] * units)

    return activation(tf.matmul(input, w) + b)

def create_monotone_dense_layer(input, units, activation, positive_transform, bias_initializer = None,
                                clip_min=None, clip_max=None):
    w = create_positive_weights(shape=[input.shape[-1].value, units], positive_transform=positive_transform)

    if bias_initializer is None:
        b = create_bias(units, clip_max=clip_max,clip_min=clip_min)
    else:
        b = create_bias(initializer=[bias_initializer] * units, clip_max=clip_max,clip_min=clip_min)

    res = tf.matmul(input, w) + b
    return res if activation is None else activation(res)

def create_cdf_layer_partial_monotonic_MLP(x_batch, y_batch, arch, positive_transform, final_activation=tf.nn.sigmoid, activation=tf.nn.tanh, print_prefix=""):
    """
    CDF(y|x)
    :param x_batch:
    :param y_batch:
    :param arch:
    :return:
    """
    with tf.name_scope("cdf_layer"):
        if x_batch is None:
            layer = y_batch
        else:
            layer = tf.concat([x_batch, y_batch], axis=1)

        layer = create_partially_monotone_dense_layer(layer, arch[0], 0 if x_batch is None else x_batch.shape[-1].value,
                                                      y_batch.shape[-1].value,
                                                      activation=activation, positive_transform=positive_transform)

        for i, size in enumerate(arch[1:]):
            with tf.variable_scope("l%d" % (i + 1)):
                layer = print_tensor(create_monotone_dense_layer(layer, size, activation=activation,
                                                    positive_transform=positive_transform), name=print_prefix+"monotone_%d"%i)
    with tf.variable_scope("cdf"):
        cdf = print_tensor(create_monotone_dense_layer(layer, 1, activation=final_activation, positive_transform=positive_transform),
                           name=print_prefix + "monotone_final")

    return cdf

def density_estimator(params, positive_transform, x_transform, y, y_size):
    pdfs = []
    cdfs = []

    activation = get_activation(params)

    for i in range(y_size):
        y_margin = tf.slice(y, [0, i], size=[-1, 1])
        with tf.variable_scope("marginal_%d" % i):
            cdfs.append(constrain_cdf(
                create_cdf_layer_partial_monotonic_MLP(x_transform, y_margin, arch=params['arch2'],
                                                       positive_transform=positive_transform, activation=activation)))
            pdfs.append(tf.maximum(create_pdf_layer(cdfs[i], y_margin), 1e-24))
    return cdfs, pdfs