#
#	Variational Autoencoder
#
#	Authors: David, Majd, Paul, Fredrick
#

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.nn as nn
import numpy as np

from utils import nn
from models.model import Model


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#			AUTOENCODER
#
#	__Note__: This is an intial setup of an autoencoder.
# 	 The parameters are static and based on dimensions of
#	 MNIST-dataset. The parameters will be adjusted to work
#	with any dataset in the end.
#
#	q_sigma and the third layer in the original decoder
# 	is defined with NonLinear in utils.nn. This is an attempt
#	to implement these layers with Tensorflow.
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class VAE(Model):

    def __init__(self, args):
        super(VAE, self).__init__(args)

        prod_input_size = np.prod(self.args.input_size)

        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.Dense(
                300, 
                input_shape=(prod_input_size,), 
                activation='relu', 
                name='EncLayer1'
            ),
            keras.layers.Dense(
                300, 
                input_shape=(300,), 
                activation='relu', 
                name='EncLayer2'
            ),
        ])

        # Latent variables
        #mean
        self.q_mean = keras.layers.Dense(
                    self.args.z1_size, 
                    input_shape=(300,), 
                    name = 'latent_mean'
                )

        #variance
        #? The researchers used an activation function to force the output
        #? of this layer to be between -6 and 2
        self.q_logvar = keras.layers.Dense(
                self.args.z1_size, 
                input_shape=(300,), 
                activation=nn.hardtanh(min_value=-6., max_value=2.).hardtanh_function, 
                name = 'latent_logvariance'
            )

        #Three layered decoder. Input is a sample z, and the decoder returns a (784,) vector.
        #This process resembles p(x | z) in the graphical representation.
        self.decoder = keras.Sequential([
            keras.layers.Dense(
                300, 
                input_shape=(self.args.z1_size, ), 
                activation='relu',
                name='DecLayer1'
            ),
            keras.layers.Dense(
                300, 
                input_shape=(300,), 
                activation='relu', 
                name='DecLayer2'
            ),
        ])
        
        self.p_mean = keras.layers.Dense(
                prod_input_size,
                input_shape=(300,), 
                activation ='sigmoid', 
                name='dec_output_mean'
        )

        if self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_logvar = keras.layers.Dense(
                prod_input_size,
                input_shape=(300,),
                activation = nn.hardtanh(min_value = -4.5, max_value = 0).hardtanh_function,
                name = 'dec_output_logvar'
            )
        
        # weight initialization
        #! Consider changing the weight initialization if necessary
        
        #TODO: add pseudoinputs if Vamprior is used
        if self.args.prior == 'Vamprior': 
            pass
    

    def q(self, x):
        """ 
        ##
        ##	Variational posterior
        ##
        ##
        ##	Input:		x			data point
        ##
        ##	Retruns 	q_z_mean	mean
        ##				q_z_var		variance
        ##
        """

        x = self.encoder(x)
        q_mean = self.q_mean(x)
        q_logvar = self.q_logvar(x)

        return q_mean, q_logvar

    

    def p(self, z):
        """ 
        ##	Generative posterior
        ##
        ##	OBS: Assume that input_type is 'binary', to begin.
        ##
        ##	Input:		z			sample point
        ##
        ##	Retruns 	x_mean		mean, which could be interpreted
        ##							as the reconstructed image.
        ##				x_var		variance. This term is only computed for
        ##							non-binary input types.
        """

        z = self.decoder(z)
        x_mean = self.p_mean(z)

        if self.args.input_type == 'binary':
            x_logvar = 0.

        else:
        #? For mom bimary data, the authors force the data to be between  0.+1./512.
        #? and 1.-1./512.
            x_mean = nn.hardtanh(min_value = 0.+1./512, max_value = 1.-1./512.).hardtanh_function(x_mean)
            x_logvar = nn.p_logvar(z)

        return x_mean, x_logvar


    def prior(self, z):
        """ 
        # ##
        # ##	Computes the log prior, p(z).
        # ##
        # ##	Inputs:	z		Samples
        # ##	
        # ##    The type of prior to compute is in 
        # ##    the attribute self.args.prior(given when running the experiment)
        # ##    It can be Gaussian, VampPrior, ...
        # ##
        """

        if self.args.prior == 'gaussian':
            res = -0.5 * tf.pow(z, 2)
            res = tf.math.reduce_sum(res)
            return res
        elif self.args.prior == 'VampPrior':
            pass
        else:
            pass


    def forwardPass(self, ):
        pass


