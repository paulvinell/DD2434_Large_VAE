#
#	Variational Autoencoder
#
import tensorflow as ts
import numpy as np
import math 

##
##
##
def p(x):
	pass

##
##
##
def q(x):
	pass

##
##	Loss function
##
def loss():
	RL = 0
	KL = 0
	pass

##
##	Computes the reparameterization trick. 
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
##	Encodes data points by computing,
##	q(z|x)
##
##	Inputs:	x 	Datapoint(-s)
##
##	Return: z	Encoded data points	
##
def encoder(x):
	pass

##
##	Decodes data points by computing,
##	p(x|z)
##
##	Inputs:	sample	A distribution sample
##
##	Return 	x_hat	Decoded 
##
def decoder(sample):
	pass


def main(dataset):
	pass