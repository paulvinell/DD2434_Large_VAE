import numpy as np
import tensorflow as tf
from tensorflow import keras

class Model(keras.Model):

    def __init__(self, args):

        super(Model,self).__init__()
        self.args = args

    def add_pseudoinputs(self):
        """ To add pseudoinputs to the Vamprior model
        """
        nonlinearity = nn.hardtanh(min_value=0., max_value=1.).hardtanh_function
        self.means = keras.layers.Dense(self.args.number_components, input_shape=np.prod(self.args.input_size), activation='sigmoid') # we shouldn't use bias here

        #initialize pseudoinputs
        if self.args.use_training_data_init: # the so-called "vampprior data"
            pass
        else:
            pass

        # create an idle input for calling pseudo-inputs
        self.idle_input = None

        return

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

    def call(self, inputs, training, mask):
        return 
