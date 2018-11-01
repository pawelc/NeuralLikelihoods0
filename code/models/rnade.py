import tensorflow as tf
import tensorflow_probability as tp

from models.utils import metric_loglikelihood, print_tensor, train_op, add_all_summary_stats, extract_xy


def rnade_model(features, labels, mode, params):
    k_mix = params["k_mix"]
    hidden_units = params['hidden_units']
    learning_rate = params["learning_rate"]
    components_distribution_param = params["components_distribution"]

    x_size, y_size, x, y = extract_xy(features, params)

    c = print_tensor(add_all_summary_stats(tf.get_variable(name="c", shape=(hidden_units,), dtype=tf.float32)))
    W = print_tensor(add_all_summary_stats(tf.get_variable(name="W", shape=(x_size + y_size-1, hidden_units), dtype=tf.float32)), "W")
    rho = add_all_summary_stats(tf.get_variable(name="rho", shape=(y_size,), dtype=tf.float32))

    if x is not None:
        a = print_tensor(add_all_summary_stats(tf.add(c, tf.matmul(x, print_tensor(tf.slice(W, [0, 0], size=[x.shape[1].value, -1]), name="x_times_W")),name='a_first')))
    else:
        a = tf.fill((tf.shape(y)[0], hidden_units), 0.0)

    ll = tf.constant([0.], dtype=tf.float32)
    lls = []
    for d in range(y_size):
        psi = add_all_summary_stats(tf.multiply(rho[d], a, name="psi_%d"%d))  # Rescaling factors
        h = add_all_summary_stats(tf.nn.relu(psi, name="h_%d"%d))  # Rectified linear unit

        V_alpha = add_all_summary_stats(tf.get_variable(name="V_alpha_%d" % d, shape=(hidden_units, k_mix), dtype=tf.float32))
        b_alpha = add_all_summary_stats(tf.get_variable(name="b_alpha_%d" % d, shape=(k_mix,), dtype=tf.float32))
        z_alpha = print_tensor(add_all_summary_stats(tf.add(tf.matmul(h, V_alpha), b_alpha, name="z_alpha_%d"%d)))

        V_mu = add_all_summary_stats(tf.get_variable(name="V_mu_%d" % d, shape=(hidden_units, k_mix), dtype=tf.float32))
        b_mu = add_all_summary_stats(tf.get_variable(name="b_mu_%d" % d, shape=(k_mix,), dtype=tf.float32))
        z_mu = tf.add(tf.matmul(h, V_mu), b_mu, name="mu_%d"%d)

        V_sigma = add_all_summary_stats(tf.get_variable(name="V_sigma_%d" % d, shape=(hidden_units, k_mix), dtype=tf.float32))
        b_sigma = add_all_summary_stats(tf.get_variable(name="b_sigma_%d" % d, shape=(k_mix,), dtype=tf.float32))
        z_sigma = print_tensor(add_all_summary_stats(tf.add(tf.matmul(h, V_sigma),b_sigma, name="z_sigma_%d"%d)))

        mu = print_tensor(add_all_summary_stats(z_mu))
        sigma = print_tensor(add_all_summary_stats(tf.maximum(1e-24,tf.square(z_sigma),name="sigma_%d" % d)))

        if components_distribution_param == "normal":
            components_distribution = tp.distributions.Normal(loc=mu, scale=sigma)
        elif components_distribution_param == "laplace":
            components_distribution = tp.distributions.Laplace(loc=mu, scale=sigma)
        else:
            raise ValueError(components_distribution_param)

        mix = tp.distributions.MixtureSameFamily(
            mixture_distribution=tp.distributions.Categorical(logits=z_alpha, allow_nan_stats=False),
            components_distribution=components_distribution, allow_nan_stats=False)

        if y is not None:
            y_d = tf.slice(y, [0, d], size=[-1, 1], name="y_%d"%d)
            ll_component = mix.log_prob(tf.reshape(y_d, [-1]))
            ll = ll + ll_component

            lls.append(ll_component)

            if d < (y_size-1):
                a = add_all_summary_stats(tf.add(a, tf.matmul(y_d, tf.slice(W, [x_size + d, 0], size=[1, -1])), name="a_%d"%d))

    if y is not None:
        ll = print_tensor(add_all_summary_stats(tf.reshape(ll, [-1, 1], name="ll")))

    if mode == tf.estimator.ModeKeys.PREDICT:
        if y is not None:
            predictions = {"log_likelihood_%d"%i : ll for i,ll in enumerate(lls)}
            predictions["log_likelihood"] =  ll
        else:
            predictions = {'samples0': tf.reshape(mix.sample(1, name="sample"),[-1,1])}

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = print_tensor(-tf.reduce_mean(ll), "loss")

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {"log_likelihood": metric_loglikelihood(ll)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op(loss, learning_rate))
