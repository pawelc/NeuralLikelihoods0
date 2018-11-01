from asynch import invoke_in_process_pool, Callable
from data import DataLoader, Config
import tensorflow as tf
import tensorflow_probability as tp
import numpy as np

from utils import create_session_config


class MPGFactory:
    def __call__(self, x):
        sd1= 4
        sd2= 3
        corr = 0.7
        sd_mat = [[sd1,0],[0, sd2]]
        corr_mat = [[1, corr],[corr,1]]

        cov = np.matmul(np.matmul(sd_mat, corr_mat),sd_mat)
        distribution = tp.distributions.MultivariateNormalFullCovariance(loc=tf.concat([0.1*tf.square(x) + x - 5, 10*tf.sin(3*x)], axis=1),
                                                          covariance_matrix=tf.constant(cov, dtype=tf.float32))
        return distribution

    def __str__(self):
        return "MPG"

class SinusoidFactory:

    def __init__(self, noise):
        self.noise = noise

    def __call__(self, x):
        if self.noise == "normal":
            return tp.distributions.Normal(loc=tf.sin(4.0 * x) + x * 0.5, scale=0.2)
        elif self.noise == "standard_t":
            return tp.distributions.StudentT(df=3.0, loc=tf.sin(4.0 * x) + x * 0.5, scale=0.2)

    def __str__(self):
        return "TrendingSinusoid"

def generate_in_tensorflow(op_factory, x_data):
    x = tf.placeholder(shape=x_data.shape,dtype=tf.float32)
    sample_op = op_factory(x).sample()
    with tf.Session(config=create_session_config()) as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(sample_op, feed_dict={x: x_data})

def compute_ll(op_factory, x_data, y_data):
    x = tf.placeholder(shape=x_data.shape, dtype=tf.float32)
    y = tf.placeholder(shape=y_data.shape, dtype=tf.float32)
    ll_op = op_factory(x).log_prob(y)
    with tf.Session(config=create_session_config()) as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(ll_op, feed_dict={x: x_data, y: y_data})

class TfGenerator(DataLoader):

    def __init__(self, conf:Config):
        super().__init__(conf)
        self.op_factory = self.conf.params["op_factory"]
        self.x_data = self.conf.params["x"]

    def generate_data(self):
        y_data = invoke_in_process_pool(1, Callable(generate_in_tensorflow, self.op_factory, self.x_data))[0]

        return np.c_[self.x_data, y_data]

    def ll(self, x_data, y_data):
        return invoke_in_process_pool(1, Callable(compute_ll, self.op_factory, x_data, y_data))[0]

    def can_compute_ll(self):
        return True








