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
@tf.function
def log_normal(x, mean, log_variance, average=False, dim=None):
	log_N = -0.5 * (log_variance + tf.pow((x - mean), 2) / tf.exp(log_variance))

	if average:
		return tf.math.reduce_mean(log_N, axis = dim)
	else:
		return tf.math.reduce_sum(log_N, axis = dim)

##
## Calculates the log probability of x according
## to a discretized log logistic distribution
##
@tf.function
def discretized_log_logistic(x, mean, logscale):
	bins = 256.
	scale = tf.exp(logscale)
	# print("#####printing mean:", mean)
	# print("#####printing scale:", scale)
	# print("#####printing x:", x)
	# x = tf.reshape(x, [x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]])
	# print("#####printing reshaped x:", x)
	x = (tf.floor(x * bins) / bins - mean) / scale
	cdf_with_x = tf.sigmoid(x + 1. / (bins * scale))
	cdf_without_x = tf.sigmoid(x)
	logp =  tf.math.log(tf.math.maximum(cdf_with_x - cdf_without_x, 1e-7))
	# take sum of logp for different images to get log likelihood of batch of 16 images
	return tf.reduce_sum(logp, 1)
