from models.nn_pdf import marginals_and_joint
from models.nn_pdf_common import transform_x, density_estimator
from models.utils import extract_xy, train_op, metric_loglikelihood
import tensorflow as tf

def nn_pdf_ar_model(features, labels, mode, params):
    positive_transform = params['positive_transform']
    learning_rate = params["learning_rate"]

    x_size, y_size, x, y = extract_xy(features, params)

    if y is None:
        raise NotImplementedError

    predictions_list = []
    all_cdfs =[]
    for y_i in range(y_size):
        with tf.variable_scope("y_%d"%y_i):
            if y_i > 0:
                y_slice = tf.slice(y, [0,y_i-1], [-1,1])
                if x is not None:
                    x = tf.concat([x, y_slice], axis=1)
                else:
                    x = y_slice
            y_marginal = tf.slice(y, [0, y_i],[-1, 1])
            x_transform = transform_x(params, x)
            cdfs, pdfs = density_estimator(params, positive_transform, x_transform, y_marginal, y_size=1)
            all_cdfs.extend(cdfs)
            predictions_list.append(marginals_and_joint(None, cdfs, pdfs, x,params))

    ll = tf.add_n([predictions["log_likelihood"] for predictions in predictions_list])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"log_likelihood_%d"%i : pred["log_likelihood"] for i,pred in enumerate(predictions_list)}
        for i,cdf in enumerate(all_cdfs):
            predictions['cdf_%d'%i] = cdf
        predictions["log_likelihood"] = ll
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.negative(tf.reduce_mean(ll), name="loss")

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {"log_likelihood": metric_loglikelihood(ll)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op(loss, learning_rate))


