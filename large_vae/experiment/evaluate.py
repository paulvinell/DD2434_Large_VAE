import time

from large_vae.experiment.train import one_pass
from large_vae.utils.visual import plot_images

import tensorflow as tf


def evaluate_model(model, train_dataset, test_dataset, args):
    """ Evaluate the best model on the evaluation dataset.
        Plot some metrics and generate images.
    """

    # We are going to test model on the first batch of the test dataset


    # Plot images
    # No need to plot all the images of the dataset, we gonna plot just 16 images
    # (i.e. args.batch_size images)
    tf.print("Plotting images")
    first_batch_test = test_dataset[0:16]
    plot_images(first_batch_test, 'real test dataset', args)

    tf.print("Plotting images (reconstructed)")
    # Reconstructing the same 16 images from the test dataset
    reconstructed_dataset = model.reconstruct_x(first_batch_test).numpy()
    # Plotting the reconstructed dataset
    plot_images(reconstructed_dataset,'reconstructed test dataset', args)

    # Generating an image
    tf.print("Plotting images (generated)")
    generated_data = model.generate_x()[0].numpy()
    plot_images(generated_data, 'generated dataset', args)


    # CALCULATE lower-bound
    tf.print("Calculating lower bound")
    t_ll_s = time.time()
    elbo_test = model.lowerBound(test_dataset)
    t_ll_e = time.time()
    print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

    t_ll_s = time.time()
    elbo_train = model.lowerBound(train_dataset)
    t_ll_e = time.time()
    print('Train lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

    # CALCULATE log-likelihood
    tf.print("Calculating log-likelihood")

    t_ll_s = time.time()
    log_likelihood_test = model.loglikelihood(test_dataset)
    t_ll_e = time.time()
    print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

    t_ll_s = time.time()
    log_likelihood_train = model.loglikelihood(train_dataset)
    t_ll_e = time.time()
    print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))


    return log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
