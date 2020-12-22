#
#	Variational Autoencoder
#
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#			Helper Functions
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

##
##	Computes p(x|z)
##
def p(x):
	pass

##
##	Computes the variational posterior, q(z| x)
##
def q(x):
	pass

##
##	Loss function
##
##	Inputs:		x 		Data point(-s)
##
##	Return: 	Loss
##				RL		Reconstruction loss
##				KL		KL-divergence
def loss(x):

	loss = 0
	RL = 0
	KL = 0

	return loss, RL, KL

##
##	Computes the reparameterization trick.
##
##	This function is called when computing the variational posterior.
##
def repTrick():
	pass

##
##	Computes the prior, p(z).
##
##	Inputs:	z		Samples
##			type 	Type of prior to compute.
##					E.g. Gaussian, VampPrior
##
def prior(z, pType='gaussian'):

	if pType == 'gaussian':
		res = -0.5 * tf.pow(z, 2)
		res = tf.math.reduce_sum(res)
		return res
	elif pType == 'VampPrior':
		pass
	else:
		pass


def forwardPass():
	pass


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#			MODEL
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

##
##	Encodes data points by computing.
##
##	Inputs:	x 					Datapoint(-s)
##
##	Return: z = (mu, sigma)		Encoded data points
##
def encoder(x):
	pass

##
##	Decodes data points by computing.
##
##	Inputs:	sample	A distribution sample
##
##	Return 	x_hat	Decoded data points
##
def decoder(sample):
	pass


def main():
	"""
	dataset = tf.keras.datasets.mnist
	(train_x, train_y), (test_x, test_y) = dataset.load_data()
	print(train_x.describe())
	"""
	pass

main()
