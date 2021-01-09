import tensorflow as tf
import tensorflow.keras as keras

#=======================================================================================================================
# ACTIVATIONS
#=======================================================================================================================

def hardtanh(min_value, max_value):
    @tf.function
    def hardtanh_(x):
        x = tf.math.maximum(x, min_value)
        x = tf.math.minimum(x, max_value)
        return x

    return hardtanh_
