#
#	Variational Autoencoder
#
#	Authors: David, Majd, Paul, Fredrick
#

import tensorflow.keras as keras
import tensorflow.nn as nn
import tensorflow as tf
import numpy as np
import math

from large_vae.utils import nn
from large_vae.utils.distributions import discretized_log_logistic, log_normal
from large_vae.models.model import Model
from scipy.special import logsumexp

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
            # keras.layers.Flatten(), # Converts (width, height, 1) -> (width*height*1)
            keras.layers.Dense(
                args.max_layer_size,
                input_shape=(prod_input_size,),
                activation='relu',
                name='EncLayer1'
            ),
            keras.layers.Dense(
                args.max_layer_size,
                input_shape=(args.max_layer_size,),
                activation='relu',
                name='EncLayer2'
            ),
        ])

        # Latent variables
        #mean
        self.q_mean = keras.layers.Dense(
                    self.args.z1_size,
                    input_shape=(args.max_layer_size,),
                    activation=nn.hardtanh(min_value=-6., max_value=2.).hardtanh_function,
                    name = 'latent_mean'
                )

        #variance
        #? The researchers used an activation function to force the output
        #? of this layer to be between -6 and 2
        self.q_logvar = keras.layers.Dense(
                self.args.z1_size,
                input_shape=(args.max_layer_size,),
                activation=nn.hardtanh(min_value=-6., max_value=2.).hardtanh_function,
                name = 'latent_logvariance'
            )

        #Three layered decoder. Input is a sample z, and the decoder returns a (784,) vector.
        #This process resembles p(x | z) in the graphical representation.
        self.decoder = keras.Sequential([
            keras.layers.Dense(
                args.max_layer_size,
                input_shape=(self.args.z1_size, ),
                activation='relu',
                name='DecLayer1'
            ),
            keras.layers.Dense(
                args.max_layer_size,
                input_shape=(args.max_layer_size,),
                activation='relu',
                name='DecLayer2'
            ),
        ])

        self.p_mean = keras.layers.Dense(
                prod_input_size,
                input_shape=(args.max_layer_size,),
                activation ='sigmoid',
                name='dec_output_mean'
        )

        self.p_logvar = keras.layers.Dense(
            prod_input_size,
            input_shape=(args.max_layer_size,),
            activation = nn.hardtanh(min_value = -4.5, max_value = 0.0).hardtanh_function,
            name = 'dec_output_logvar'
        )

        # weight initialization
        #! Consider changing the weight initialization if necessary

        # add pseudoinputs if Vamprior is used
        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

    @tf.function
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
        # print("############ printing x: ", x)
        # print("############ tf.shape(x): ", tf.shape(x))
        q_logvar = self.q_logvar(x)

        return q_mean, q_logvar


    @tf.function
    def p(self, z):
        """
        ##	Generative posterior
        ##
        ##
        ##	Input:		z			sample point
        ##
        ##	Retruns 	x_mean		mean, which could be interpreted
        ##							as the reconstructed image.
        ##				x_var		variance.
        """

        z = self.decoder(z)
        x_mean = self.p_mean(z)

        #? For non bimary data, the authors force the data to be between  0.+1./512.
        #? and 1.-1./512.
        x_mean = nn.hardtanh(min_value = 0.+1./512, max_value = 1.-1./512.).hardtanh_function(x_mean)
        x_logvar = self.p_logvar(z)

        return x_mean, x_logvar


    @tf.function
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
            # print("----------------> res: ", res)
            return res

        # In this part we only evaluate the VampPrior given the current pseudoinputs; we do not learn the pseudoinputs
        elif self.args.prior == 'vampprior':
            pseudoinputs = self.args.pseudoinput_count

            # The idle input is used to evaluate the pseudoinputs
            # The idle input has input size (pseudoinput_count) and output size (the image dimensions, e.g. 784)
            U = self.means(self.idle_input) # defined in model.py

            # Here we use the learned variational distribution q to infer the posterior that is our vampprior
            z_p_mean, z_p_logvar = self.q(U) # dimensions: pseudoinputs x M

            # Expand the argument to this function and the inferred
            z_expand = tf.expand_dims(z,1)
            means = tf.expand_dims(z_p_mean,0)
            logvars = tf.expand_dims(z_p_logvar,0)

            density = log_normal(z, means, logvars, dim=2) - math.log(pseudoinputs) # p_lambda, dimensions: batch size x pseudoinputs
            density_max = tf.math.reduce_max(density, axis=1) # dimensions: batch size x 1

            log_p_z = density_max + tf.log(tf.reduce_sum(tf.exp(density - tf.expand_dims(density_max,1)),1)) # dimensions: batch size x 1

            return log_p_z

        else:
            pass

    @tf.function
    def forwardPass(self, x):
        """
        # ##
        # ##    Uses data point and traverse one time through the network.
        # ##
        # ##    Inputs:     x            Data points, such as images.
        # ##
        # ##    Returns:    x_mean       Mean values of the reconstructed data point.
        # ##                x_logvar     Variance of the reconstructed data point.
        # ##                z            Sample
        # ##                z_mean       Mean of encoded data points.
        # ##                z_logvar     Variance of encoded data points.
        # ##
        """
        z_mean, z_logvar = self.q(x)
        # print('----------> z_mean:', z_mean)
        # print('----------> z_logvar:', z_logvar)
        z = self.repTrick(z_mean, z_logvar)
        # print('----------> z:', z)
        x_mean, x_logvar = self.p(z)

        return x_mean, x_logvar, z, z_mean, z_logvar

    @tf.function
    def loss(self, x, beta=1., average=False):
        """
        # ##
        # ##    Loss Function.
        # ##
        # ##    Inputs:     x          Data points, such as images.
        # ##                beta       Cost minimizing parameter
        # ##
        # ##
        # ##    Outputs:    loss       Difference between true and predicted value
        # ##                RE         Reconstruction error
        # ##                KL         Regularizing value
        # ##
        """
        #One pass through the network.
        x_mean, x_logvar, z, z_mean, z_logvar = self.forwardPass(x)

        ##
        ##  Reconstruction error
        ##
        RE = discretized_log_logistic(x, x_mean, x_logvar) # p(x|z)

        log_prior = self.prior(z) # log p(z)
        log_q_z = log_normal(z, z_mean, z_logvar, dim=1) # Our learned variational distribution, log q(z|x).
        # print('------------> log_prior:', log_prior)
        # print('------------> log_q_z:', log_q_z)
        KL = log_q_z - log_prior
        # print("##### KL:", KL)
        # print("##### beta:", beta)
        # print("##### RE:", RE)
        loss = KL * beta - RE

        if average:
            return tf.reduce_mean(loss), tf.reduce_mean(RE), tf.reduce_mean(KL)
        else:
            return loss, RE, KL


    def loglikelihood(self, x, sample_size=1, batch_size=32):
        """
        # ##
        # ##    Estimate the marginal log likelihood
        # ##
        # ##    Inputs:     sample_size: the number of sample points for importance sampling
        # ##                batch_size: the size of the batch for calculating the loss
        # ##
        # ##    Returns:    loglikelihood
        # ##
        """
        test_size = x.shape[0] # get number of rows in test data

        likelihood_test = []

        if sample_size <= batch_size:
            rounds = 1
        else:
            rounds = sample_size / batch_size
            sample_size = batch_size

        for i in range(test_size):
            x_data_point = tf.expand_dims(x[0],0) # wrap single row in brackets

            losses = []
            for r in range(0, int(rounds)):
                # Copy data point to get # of rows == sample_size
                tf.broadcast_to(x_data_point, [sample_size, x_data_point.shape[1]])

                loss_for_data_point, _, _ = self.loss(x) # self.loss should not average before returning

                losses.append(-loss_for_data_point)

            # Calculate max
            losses = np.asarray(losses)
            #  Reshape into the form
            #  array([[1],
            #   [2],
            #   [3],
            #   ...,
            #   [sample_size]])
            losses = np.reshape(losses, (losses.shape[0] * losses.shape[1], 1))
            likelihood_x = logsumexp(losses)
            likelihood_test.append(likelihood_x - np.log(len(losses)))

        likelihood_test = np.array(likelihood_test)

        return -np.mean(likelihood_test)

    def lowerBound(self, X, MB = 100):
        """
        # ##
        # ##    Computes the lower bound, which is part of the evaluation
        # ##    process.
        # ##
        # ##    Inputs:     X       All data points
        # ##                MB      Minibatches
        # ##
        # ##    Returns:    LB      Lower Bound value
        # ##
        """

        LB = 0.
        RE_tot = 0.
        KL_tot = 0.

        I = int(math.ceil(X.shape[0]/ MB))

        for i in range(I):

            x = X[i * MB: (i+1)*MB]                             #####
            x = tf.reshape(x, (-1,np.prod(self.args.input_size)))    #####

            loss, RE, KL = self.loss(x, average=True)

            LB += tf.identity(loss).numpy()
            RE_tot += tf.identity(RE).numpy()
            KL_tot += tf.identity(KL).numpy()

        return LB/I

    @tf.function
    def generate_x(self, N=16):
        """
        # ##
        # ##    Generates some z-samples from a distribution. Samples will be
        # ##    passed through the decoder to evaluate the model.
        # ##
        # ##    Inputs:     N               Number of samples to generate
        # ##
        # ##    Returns:    sample_rand     Samples
        # ##
        """
        if self.args.prior == 'gaussian':
            z_sample_rand = tf.random.normal(
                [N, self.args.z1_size]
            )

        elif self.args.prior == 'vampprior':
            means = tf.slice(self.means(self.idle_input), [0],[N]) # check this, self.means and self.idle_input are defined in model.py
            z_sample_gen_mean, z_sample_gen_logvar = self.q(means)
            z_sample_rand = self.repTrick(z_sample_gen_mean, z_sample_gen_logvar)

        sample_rand = self.p(z_sample_rand)

        return sample_rand

    @tf.function
    def reconstruct_x(self,x):
        """
        # ##
        # ##    Reconstructs an image, by passing it through the network.
        # ##
        # ##    Inputs:     X               Data points
        # ##
        # ##    Returns:    x_mean          Reconstructed data point, e.g. an image.
        # ##
        """
        x_mean, _, _, _, _ = self.forwardPass(x)
        return x_mean

    @tf.function
    def call(self, inputs, training, mask):
        return self.forwardPass(inputs)
