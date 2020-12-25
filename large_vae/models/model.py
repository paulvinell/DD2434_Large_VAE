import tensorflow as tf
from tensorflow import keras


class Model(keras.Model):

    def __init__(self,args):

        super(Model,self).__init__()
        self.args = args

    def add_pseudoinputs(self):
        """ To add pseudoinputs to the Vamprior model
        """
        return 

    def repTrick(mu, logvar):
        """ Reparameterization trick
        """ 

        eps = tf.random.normal(tf.shape(mu), mean=0, stdev=1)
        # Note: e^(logvar * 0.5) = sqrt(variance) = standard deviation
        res = mu + tf.exp(logvar * 0.5) * eps

        return res
