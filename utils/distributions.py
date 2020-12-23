import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.nn as nn
import numpy as np
import scipy
import math

#Global
minEps = 1e-5
maxEps = 1.-1e-5

##
##	Gaussian function in logarithmic scale
##
def normal(x, mean, variance, average=False, dim=None):
	log_N = -0.5 * (variance + tf.pow((x - mean), 2) / tf.exp(variance))

	if average:
		return tf.math.reduce_mean(log_N, axis = dim)
	else:
		return tf.math.reduce_sum(log_N, axis = dim)

##
##	Log Bernoulli 
##
def bernoulli(x, x_hat, average=False, dim=None):
	
	probabilities = tf.clip_by_value(
    x_hat, clip_value_min = minEps, clip_value_max= maxEps, 
	)

	bern = x * tf.log(probabilities) + (1. -x) * tf.log(1. - probabilities)

	if average:
		return tf.math.reduce_mean(bern, dim)
	else:
		return tf.math.reduce_sum(bern, axis=dim)


