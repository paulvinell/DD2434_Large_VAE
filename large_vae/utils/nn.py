import tensorflow as tf
import tensorflow.keras as keras

#=======================================================================================================================
# ACTIVATIONS
#=======================================================================================================================

# HardTanh activation
class hardtanh(keras.Model):
    def __init__(self, min_value = -1.0, max_value = 1.0):
        super(hardtanh, self).__init__()
        self.min_value = min_value
        self.max_value = max_value


    def hardtanh_function(self, tensor):
        def hardtanh_(x):
            if x <= self.min_value:
                return self.min_value
            elif x >= self.max_value:
                return self.max_value
            else:
                return x


        def apply_on_line(tensor_line):
            return tf.map_fn(hardtanh_, tensor_line)
        

        return tf.map_fn(apply_on_line, tensor)
