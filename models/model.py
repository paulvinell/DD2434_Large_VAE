import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model as md

# Optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)

class Model(md):

    def __init__(self, args):

        super(Model,self).__init__()
        self.args = args

    def add_pseudoinputs(self):
        """ To add pseudoinputs to the Vamprior model
        """
        return

    ##
    ##	Reparameterization trick.
    ##
    ## Inputs:	mu
    ##			logvar
    ##
    ## Return:	reparameterization
    def repTrick(mu, logvar):
        """ Reparameterization trick
        """

        eps = tf.random.normal(tf.shape(mu), mean=0, stdev=1)
        # Note: e^(logvar * 0.5) = sqrt(variance) = standard deviation
        res = mu + tf.exp(logvar * 0.5) * eps

        return res


    """ Performs a training step for a set of samples """
    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
