# Thanks to Elvis Dohmatob for the original loading script of Frey Faces
# His blog post: https://dohmatob.github.io/research/2016/10/22/VAE.html
# Differences: I changed from Python 2 -> Python 3

import os
from urllib.request import urlopen, URLError, HTTPError
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import tensorflow as tf


def preprocess(data):
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1)) / 255.0
    return data.astype('float32')


def fetch_file(url):
    try:
        f = urlopen(url)
        print("Downloading data file " + url + " ...")

        # Open our local file for writing
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())
        print("Done.")

    #handle errors
    except HTTPError as e:
        print("HTTP Error:", e.code, url)
    except URLError as e:
        print("URL Error:", e.reason, url)

def load_frey_faces():
    url =  "http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat"
    data_filename = os.path.basename(url)
    if not os.path.exists(data_filename):
        fetch_file(url)
    else:
        print("Data file %s exists." % data_filename)

    # reshape data for later convenience
    img_rows, img_cols = 28, 20
    ff = loadmat(data_filename, squeeze_me=True, struct_as_record=False)
    ff = ff["ff"].T.reshape((-1, img_rows, img_cols))

    num_data_points = ff.shape[0]
    num_training = int(num_data_points * 0.9)
    train_x = ff[:num_training,:,:] # Training data
    test_x = ff[num_training:,:,:] # Testing data

    return train_x, test_x


def load_mninst_dataset():
    dataset = tf.keras.datasets.mnist
    ((train_x, train_y), (test_x, test_y)) = dataset.load_data() # I don't think we need the labels

    return train_x, test_x


def load_experiment_dataset(args):
    """Load the dataset chosen to perform the experiment
    """
    if args.dataset == 'mnist':
        train_x, test_x = load_mninst_dataset()
    elif args.dataset == 'frey':
        train_x, test_x = load_frey_faces()

    train_x, val_x = train_test_split(train_x, test_size=args.evaluation)

    train_x = preprocess(train_x)
    val_x = preprocess(val_x)
    test_x = preprocess(test_x)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_x).shuffle(train_x.shape[0]).batch(args.batch_size)
    eval_dataset = tf.data.Dataset.from_tensor_slices(val_x).shuffle(val_x.shape[0]).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_x).shuffle(test_x.shape[0]).batch(args.batch_size)

    return train_dataset, eval_dataset, test_dataset
