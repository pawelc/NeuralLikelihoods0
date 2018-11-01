import tensorflow as tf
import functools as func
import tensorflow_probability as tp

if __name__ == '__main__':

    # cov = tf.constant([[1e-10,0.756036699],[0.756036699,0.784856319]])
    cov = tf.constant([[8.9,7.9],[3.0, 2.0]])

    # det=tf.matrix_determinant(cov)
    e , v = tf.self_adjoint_eig(cov)

    def adjust_cov(cov,eig_val):
        eig_val = tf.maximum(1e-6,tf.abs(eig_val))
        cov1 = cov + tf.eye(2)*tf.abs(eig_val)
        cov1_print = tf.Print(cov1,[cov, eig_val, cov1], summarize=1000, message="adjust cov")
        e, v = tf.self_adjoint_eig(cov1_print)
        ePrint = tf.Print(e[0], [e[0]], message="Eigen: ", summarize=1000)
        return cov1_print, ePrint

    def cond(cov,eig_val):
        return tf.less_equal(eig_val, 1e-12)

    cov,_ = tf.while_loop(cond, adjust_cov, [cov, e[0]])

    # cov = tf.cond(e[0] < 0, adjust_cov, pure_cov)

    # inv = tf.matrix_inverse(a)
    c_pdf = tp.distributions.MultivariateNormalFullCovariance(loc=tf.constant([0.0] * 2),
                                                              covariance_matrix=cov)
    prob = c_pdf.prob([[0.,0.]])
    # res = tf.cond(tf.greater(det, 1e-6), lambda: tf.matrix_inverse(a),lambda: tf.matrix_inverse(a + 1e-5*tf.eye(2)))

    with tf.Session() as sess:
        print(sess.run(prob))