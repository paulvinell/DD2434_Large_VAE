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
        #? For non bimary data, the authors force the data to be between  0.+1./512.
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
        elif self.args.prior == 'vampprior':
            # TODO: stuff
			q_mean, q_sigma = self.q(z)
			# TODO: other stuff
        else:
            pass


    def forwardPass(self, x):
        """
        # ##
        # ##    Uses data point and traverse one time through the network.
        # ##
        # ##    Inputs:     x            Data points, such as images.
        # ##
        # ##    Returns:    x_mean       Mean values of the reconstructed data point.
        # ##                x_logvar     Variance of the reconstructed data point.
        # ##                             Observe. This is only availabe for non-binary
        # ##                             input types.
        # ##                z            Sample
        # ##                z_mean       Mean of encoded data points.
        # ##                z_logvar     Variance of encoded data points.
        # ##
        """
        z_mean, z_logvar = self.q(x)
        z = self.repTrick(z_mean, z_logvar)
        x_mean, x_logvar = self.p(z)

        return x_mean, x_logvar, z, z_mean, z_logvar


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
        # ##                RE         Log-Likelihood
        # ##                KL         Regularizing value
        # ##
        """
        #One round through the network.
        x_hat, x_var, z_sample, q_mean, q_var = self.forwardPass(x)

        ##
        ##  Log-likelihood. Computes P(z|x).
        ##
        #-----------------------------------------
        #TODO: This section requires the LL-part
        #      to compute the likelihood.
        #-----------------------------------------
        if self.args.input_type == 'binary':
            RE = bernoulli(x, x_hat, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = 0  #OBS. This is a temporary setting.
            #RE = loglikelihood(X) # p(z|x)
        else:
            raise Exception('Data type is not supported.')
        #-----------------------------------------
        #       END
        #-----------------------------------------

        ##
        ##  Regularizer. Computes KL(q(z|x) || p(z)).
        ##
        q_z = normal(z_sample, q_mean, q_var, dim=1) #Our learned distribution, q(z|x).
        prior = self.prior(x)   #Prior, p(z)
        KL = -(q_z - prior)

        loss = KL*beta - RE

        if average:
            return tf.reduce_mean(loss), tf.reduce_mean(RE), tf.reduce_mean(KL)
        else:
            return loss, RE, KL


    def loglikelihood(self, X, sample_size=5000, batch_size=100):
        ''' Estimate the marginal log likelihood using importance sampling
        @param sample_size: the number of sample points for importance sampling
        @param batch_size: the size of the batch for calculating the loss

        '''
        if sample_size <= batch_size:
            rounds = 1
        else:
            rounds = sample_size / batch_size
            sample_size = batch_size

            x_data_point = X.expand_dims(0)

        losses = []
        for r in range(0, int(rounds)):
            # Repeat for all data points
            x = x_data_point.expand(sample_size, x_data_point.size(1))

            loss_for_data_point, _, _ = self.calculate_loss(x)

            losses.append(-loss_for_data_point)

        # Calculate max using logsumexp
        losses = np.asarray(losses)
        losses = np.reshape(losses, (losses.shape[0] * losses.shape[1], 1))
        likelihood_x = tf.math.reduce_logsumexp(losses)
        # Calculate log mean
        return likelihood_x - np.log(len(losses))


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

        I = int(math.ceil(X.size[0] / MB))

        for i in range(I):

            x = X[i * MB: (i+1)*MB].view(-1,np.prod(self.args.input_size))

            loss, RE, KL = self.loss(x, average=True)

            LB += loss.cpu().data
            RE_tot += RE.cpu().data
            KL_tot += KL.cpu().data

        return LB/I


    #? This function is used by the authors in the evaluation module but
    #? I don't understand why for now.
    #? I code it just in case we need it
    def generate_x(self, N=25):
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
        if self.args.prior == 'standard':
            z_sample_rand = tf.Variable(tf.random.normal(
                [N, self.args.z1_size]
            ))

        elif self.args.prior == 'vampprior':
            #TODO: When we will implement VampPrior
            pass

        sample_rand = self.p(z_sample_rand)

        return sample_rand


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
