#
#	Variational Autoencoder
#
import tensorflow as ts
import numpy as np
import math 

##
##	Computes p(x|z)
##
def p(x):
	pass

##
##	Computes the variational posterior, q(z | x)
##
def q(x):
	pass

##
##	Loss function
##
##	Inputs:		x 		Data point(-s)
##
##	Return: 	Loss 	 	
##				RL
##				KL
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
def reparTrick():
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
		pass
	elif pType == 'VampPrior':
		pass
	else:
		pass

##
##	Encodes data points by computing.
##
##	Inputs:	x 	Datapoint(-s)
##
##	Return: z	Encoded data points	
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


def main(dataset):
	pass
