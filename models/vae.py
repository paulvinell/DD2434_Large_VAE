#
#	Variational Autoencoder
#
#	Authors: David, Majd, Paul, Fredrick
#
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.nn as nn
import numpy as np
import scipy
import math

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

		if args.gaussian:
		    pass
		elif args.mog:
		    pass
		elif args.vamp:
		    pass

		if args.vae:
		    pass
		elif args.hvae:
		    pass

		if args.mnist:
		    self.inputSize = [1, 28, 28]
		elif args.frey:
		    self.inputSize = [1, 20, 28]

		prod_input_size = np.prod(self.inputSize)

		self.encoder = keras.Sequential([
			keras.layers.Dense(
				300,
				input_shape=(np.prod(self.inputSize)	,),
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

		# TODO: fix this
		self.z1_size = 1

		#mean
		self.q_mean = keras.Sequential(
			keras.layers.Dense(self.z1_size, input_shape=(300,))
		)

		#variance
		self.q_var = keras.Sequential(
			keras.layers.Dense(self.z1_size, input_shape=(300,), activation='tanh')
		)

		#Three layered decoder. Input is a sample z, and the decoder returns a (784,) vector.
		#This process resemles p(x | z) in the graphical representation.
		self.decoder = keras.Sequential([
			keras.layers.Dense(
				300,
				input_shape=(self.z1_size, ),
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

		# if self.args.input_type == "binary":
		# 	keras.layers.Dense(
		# 		prod_input_size,
		# 		input_shape=(300,),
		# 		activation ='sigmoid',
		# 		name='DecNonLinearLayer'
		# 	)
		# else:
		# 	#TODO: Others type of input_type
		# 	pass

	##
	##	Variational posterior
	##
	##	OBS: Assume that input_type is 'binary', to begin.
	##
	##	Input:		x			data point
	##
	##	Retruns 	q_z_mean	mean
	##				q_z_var		variance
	##
	def q(self, x):

		x = self.encoder(x)
		q_mean = self.q_mean(x)
		q_logvar = self.q_var(x)

		return q_mean, q_sigma


	##
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
	def p(self, z):

		x_mean = self.decoder(z)
		x_logvar = 0
		return x_mean, x_logvar


	##
	##	Loss function
	##
	##	Inputs:		x 		Data point(-s)
	##
	##	Return: 	Loss
	##				RL		Reconstruction loss
	##				KL		KL-divergence
	def loss(self, x):
		loss = 0
		RL = 0

		log_p_z = self.prior(x, 'gaussian')
		log_q_z = 0
		KL = -(log_p_z - log_q_z)

		return loss, RL, KL


	##
	##	Computes the log prior, p(z).
	##
	##	Inputs:	z		Samples
	##			type 	Type of prior to compute.
	##					E.g. Gaussian, VampPrior
	##
	def prior(self, z, pType='gaussian'):

		if pType == 'gaussian':
			# Derived from the logarithm of the pdf of the normal distribution.
			# Some constants have been dropped.
			res = -0.5 * tf.pow(z, 2)
			res = tf.math.reduce_sum(res)
			return res
		elif pType == 'VampPrior':
			# TODO: stuff
			q_mean, q_sigma = self.q(z)
			# TODO: other stuff
		else:
			pass


	def forwardPass(self):
		pass
