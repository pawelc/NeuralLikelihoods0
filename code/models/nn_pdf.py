import tensorflow as tf

from models.nn_pdf_common import transform_x, density_estimator
from models.utils import log_likelihood_from_cdfs_transforms, metric_loglikelihood, train_op, extract_xy

def get_loss(predictions):
    return tf.negative(tf.reduce_mean(predictions["log_likelihood"]), name="loss")

def marginals_and_joint(cov_type, cdfs, pdfs,x, params):
    predictions={}
    lls = []

    for i, cdf, pdf in zip(range(len(pdfs)), cdfs, pdfs):
        predictions["cdf%d" % i] = cdf
        predictions["pdf%d" % i] = pdf
        lls.append(tf.log(pdf))

    predictions["log_likelihood"] = log_likelihood_from_cdfs_transforms(cov_type, cdfs, lls,x, params)
    return predictions

def nn_pdf_model(features, labels, mode, params):
    positive_transform = params['positive_transform']
    learning_rate = params["learning_rate"]
    cov_type = params["cov_type"]

    x_size, y_size, x, y = extract_xy(features, params)

    if y is None:
        raise NotImplementedError

    x_transform = transform_x(params, x)

    cdfs, pdfs = density_estimator(params, positive_transform, x_transform, y, y_size)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = marginals_and_joint(cov_type, cdfs, pdfs,x, params)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    predictions = marginals_and_joint(cov_type, cdfs, pdfs, x, params)
    loss = get_loss(predictions)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {"log_likelihood": metric_loglikelihood(predictions["log_likelihood"])}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op(loss, learning_rate))