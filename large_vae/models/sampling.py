import tensorflow as tf

class Sampling(tf.keras.layers.Layer):

    @tf.function
    def call(self, inputs):
        mu, logvar = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        eps = tf.random.normal((batch, dim), mean=0.0, stddev=1.0)
        res = mu + tf.exp(logvar * 0.5) * eps
        return res
