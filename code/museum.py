import math

import tensorflow as tf

oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)  # normalisation factor for gaussian, not needed.

def tf_normal(y, mu, sigma):
    result = tf.subtract(y, mu)
    result = tf.multiply(result, tf.reciprocal(sigma))
    result = -tf.square(result) / 2
    return tf.multiply(tf.exp(result), tf.reciprocal(sigma)) * oneDivSqrtTwoPI

def custom_gaussian(x, mu, std):
    x_norm = (x - mu) / std
    result = oneDivSqrtTwoPI * math.exp(-x_norm * x_norm / 2) / std
    return result

def softmax(logits, name="softmax"):
    max_pi = tf.reduce_max(logits, 1, keepdims=True)
    # out_pi = tf.subtract(logits, max_pi)

    out_pi = tf.exp(logits)

    normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keepdims=True))
    out_pi = tf.multiply(normalize_pi, out_pi, name=name)
    return out_pi