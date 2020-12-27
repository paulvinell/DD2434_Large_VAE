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
