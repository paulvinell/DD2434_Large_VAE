#
# Notes:
# Created using Tensorflow 2.3.0
#
# Parameters:
# Dataset: --mnist or --frey
# Prior: --gaussian, --mog, or --vamp
# Model: --vae or --hvae
#
# Example initiation:
# $ python main.py --frey --mog --vae

###################
# Parse arguments #
###################

import argparse
parser = argparse.ArgumentParser()

# Datasets
parser.add_argument("--mnist", help='Dataset: MNIST', action='store_true')
parser.add_argument("--frey", help='Dataset: Frey Faces', action='store_true')

# Prior type
parser.add_argument("--gaussian", help='Prior: Gaussian', action='store_true')
parser.add_argument("--mog", help='Prior: Mixture of Gaussians', action='store_true')
parser.add_argument("--vamp", help='Prior: VAMP', action='store_true')

# Model type
parser.add_argument("--vae", help='Model: Variational Auto-Encoder', action='store_true')
parser.add_argument("--hvae", help='Model: Hierarchical Variational Auto-Encoder', action='store_true')

args = parser.parse_args() # Contains binary values for each flag

############
# Defaults #
############
if not args.mnist and not args.frey:
    args.mnist = True

if not args.gaussian and not args.mog and not args.vamp:
    args.vamp = True

if not args.vae and not args.hvae:
    args.hvae = True

##################
# Run experiment #
##################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Using Tensorflow version {}".format(tf.__version__))

if args.mnist:
    dataset = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = dataset.load_data() # I don't think we need the labels
elif args.frey:
    from load_frey_faces import load_frey_faces
    dataset = load_frey_faces()
    num_data_points = dataset.shape[0]
    num_training = int(num_data_points * 0.9)
    train_x = dataset[:num_training,:,:] # Training data
    test_x = dataset[num_training:,:,:] # Testing data

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
