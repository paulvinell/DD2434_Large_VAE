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
import time

from large_vae.utils import nn
from large_vae.utils.distributions import discretized_log_logistic, log_normal
from large_vae.models.model import Model
from large_vae.models.sampling import Sampling
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

        # Encoder
        self.encoder = self.init_encoder(args)

        #Three layered decoder. Input is a sample z, and the decoder returns a (784,) vector.
        #This process resembles p(x | z) in the graphical representation.
        self.decoder = self.init_decoder(args)

        # weight initialization
        #! Consider changing the weight initialization if necessary

        # add pseudoinputs if Vamprior is used
        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

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
            z_p_mean, z_p_logvar, _ = self.encoder(U) # dimensions: pseudoinputs x M

            # Expand the argument to this function and the inferred
            z_expand = tf.expand_dims(z,1)
            means = tf.expand_dims(z_p_mean,0)
            logvars = tf.expand_dims(z_p_logvar,0)

            density = log_normal(z_expand, means, logvars, dim=2) - tf.math.log(tf.dtypes.cast(pseudoinputs, tf.float32)) # p_lambda, dimensions: batch size x pseudoinputs
            density_max = tf.math.reduce_max(density, axis=1) # dimensions: batch size x 1

            log_p_z = density_max + tf.math.log(tf.reduce_sum(tf.exp(density - tf.expand_dims(density_max,1)),1)) # dimensions: batch size x 1

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
        z_mean, z_logvar, z = self.encoder(x)
        # print('----------> z:', z)
        x_mean, x_logvar = self.decoder(z)

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


    def loglikelihood(self, x, sample_size=128):
        """
        # ##
        # ##    Estimate the marginal log likelihood
        # ##
        # ##    Inputs:     sample_size: the number of sample points for importance sampling
        # ##                x: the dataset
        # ##
        # ##    Returns:    loglikelihood
        # ##
        """
        test_size = x.shape[0] # get number of rows in test data
        likelihood_test = np.zeros((test_size,))

        progress_update = time.time()

        for i in range(test_size): # For each image in the dataset
            x_data_point = tf.expand_dims(x[i], 0) # Get the image

            # Copy data point to get # of rows == sample_size
            x_copies = tf.broadcast_to(x_data_point, [sample_size, x_data_point.shape[1]])

            losses, _, _ = self.loss(x_copies) # self.loss should not average before returning

            ### Calculates the log of the average loss
            # losses = [log(loss_1), log(loss_2), ..., log(loss_n)]
            #
            # likelihood_x = log[exp(log(loss_1)) + exp(log(loss_2)) + ... + exp(log(loss_n))]
            #              = log[loss_1 + loss_2 + ... + loss_n]
            #              = log[n * avg_loss]
            #              = log[n] + log[avg_loss]
            #
            # likelihood_test[i] = likelihood_x - log(n)
            #                    = log[avg_loss]

            likelihood_x = logsumexp(losses)
            likelihood_test[i] = likelihood_x - np.log(sample_size)

            if time.time() - progress_update >= 10:
                print("Progress update: processed {}/{} ({:.3g}%)".format(i+1, test_size, 100*(i+1)/test_size))
                progress_update = time.time()

        return np.mean(likelihood_test)

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
            means = self.means(self.idle_input)
            means = tf.slice(means, [0,0],[(N if N < self.args.pseudoinput_count else self.args.pseudoinput_count), np.prod(self.args.input_size)]) # check this, self.means and self.idle_input are defined in model.py
            _, _, z_sample_rand = self.encoder(means)

        sample_rand = self.decoder(z_sample_rand)

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

    def init_encoder(self, args):
        prod_input_size = np.prod(self.args.input_size)

        encoder_inputs = keras.Input(shape=(prod_input_size,))
        x = keras.layers.Dense(args.max_layer_size, activation='relu', name='EncLayer1')(encoder_inputs)
        x = keras.layers.Dense(args.max_layer_size, activation='relu', name='EncLayer2')(x)
        x = keras.layers.Flatten()(x)
        q_mean = keras.layers.Dense(self.args.z1_size, activation=nn.hardtanh(-6., 2.), name = 'latent_mean')(x)
        q_logvar = keras.layers.Dense(self.args.z1_size, activation=nn.hardtanh(-6., 2.), name = 'latent_logvariance')(x)

        z = Sampling()([q_mean, q_logvar])

        encoder = keras.Model(encoder_inputs, [q_mean, q_logvar, z], name="encoder")

        return encoder

    def init_decoder(self, args):
        prod_input_size = np.prod(self.args.input_size)

        decoder_inputs = keras.Input(shape=(args.z1_size,))
        x = keras.layers.Dense(args.max_layer_size, activation='relu', name='EncLayer1')(decoder_inputs)
        x = keras.layers.Dense(args.max_layer_size, activation='relu', name='EncLayer2')(x)
        x = keras.layers.Flatten()(x)

        p_mean = keras.layers.Dense(prod_input_size, activation='sigmoid', name = 'dec_output_mean')(x)
        p_mean = keras.layers.Activation(nn.hardtanh(0.+1./512,1.-1./512.))(p_mean)

        p_logvar = keras.layers.Dense(prod_input_size, activation=nn.hardtanh(-4.5, 0.0), name = 'dec_output_logvar')(x)

        decoder = keras.Model(decoder_inputs, [p_mean, p_logvar], name="decoder")

        return decoder
