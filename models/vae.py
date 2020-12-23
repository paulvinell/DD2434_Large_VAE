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
import torch
import scipy
import math


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


#Image Dimensions (28x28), which is the dimension
#of one data point in the MINST.
inputSize = [1, 28, 28]

#Doule layered encoder. Takes one image with dimension (784,) as input, x, and returns a (300,) vector. 
#This process resemles q(z | x) in the graphical representation. 
encoder = keras.Sequential([
	keras.layers.Dense(300, input_shape=(np.prod(inputSize)	,), activation='relu', name='EncLayer1'),
	keras.layers.Dense(300, input_shape=(300,), activation='relu', name='EncLayer2'),
])

z_size = 40		# 40 stochastic hidden units for z - retrieved from the report p.5
#mean
q_mu = keras.Sequential(
	keras.layers.Dense(z_size, input_shape=(300,))
)
#variance
q_sigma = keras.Sequential(
	keras.layers.Dense(z_size, input_shape=(300,), activation='tanh')
)

#Three layered decoder. Input is a sample z, and the decoder returns a (784,) vector.
#This process resemles p(x | z) in the graphical representation. 
decoder = keras.Sequential([
	keras.layers.Dense(300, input_shape=(40, ), activation='relu', name='DecLayer1'),
	keras.layers.Dense(300, input_shape=(300,), activation='relu', name='DecLayer2'),
	keras.layers.Dense(np.prod(inputSize), input_shape=(300,), activation ='sigmoid', name='DecNonLinearLayer')
])



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
def q(x):

	x = encoder(x)
	q_mean = q_mu(x)
	q_logvar = q_sigma(x)

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
def p(z):

	x_mean = decoder(z)
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
def loss(x):

	loss = 0
	RL = 0
	KL = 0

	return loss, RL, KL

##
##	Reparameterization trick.
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

