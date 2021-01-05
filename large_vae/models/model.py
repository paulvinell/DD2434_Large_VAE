import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from large_vae.utils import nn

class Model(keras.Model):

    def __init__(self, args):

        super(Model,self).__init__()
        self.args = args

    def add_pseudoinputs(self):
        """ To add pseudoinputs to the Vamprior model
        """
        nonlinearity = nn.hardtanh(min_value=0., max_value=1.).hardtanh_function

        # initialize pseudoinputs
        if self.args.use_training_data_init: # the so-called "vampprior data"
            pass
        else:
            ps_mean, ps_stddev = self.args.pseudoinputs_mean, self.args.pseudoinputs_std
            initializer = keras.initializers.RandomNormal(mean=ps_mean, stddev=ps_stddev, seed=None)
            self.means = keras.layers.Dense(self.args.pseudoinput_count, input_shape=(np.prod(self.args.input_size),), use_bias=False, activation='sigmoid',
            kernel_initializer=initializer)

        # create an idle input for calling pseudoinputs
        self.idle_input = tf.Variable(tf.eye(self.args.pseudoinput_count, self.args.pseudoinput_count), trainable=False)

    ##
    ##	Reparameterization trick.
    ##
    ## Inputs:	mu
    ##			logvar
    ##
    ## Return:	reparameterization
    def repTrick(self, mu, logvar):
        """ Reparameterization trick
        """

        eps = tf.random.normal(tf.shape(mu), mean=0, stddev=1)
        # Note: e^(logvar * 0.5) = sqrt(variance) = standard deviation
        res = mu + tf.exp(logvar * 0.5) * eps

        return res
