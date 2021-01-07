import time
import numpy as np

from tensorflow.python.lib.io import file_io
import tensorflow as tf

from large_vae.experiment.train import one_pass
from large_vae.experiment.evaluate import evaluate_model
from large_vae.utils.visual import plot_history
from large_vae.utils.load_data import batch_data


def optimizer(lr):
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    return optimizer

def run_experiment(model, train_x, val_x, test_x, args):

    """ Run the complete experiment.
        That is, for each epoch, train the dataset and
        evaluate it.

        The model is ready when the loss function has increased
        over two successive epochs or when we reach the max number
        of epochs passed as argument in CLI.

    """

    train_dataset = batch_data(train_x, args)   # tf.data.Dataset.from_tensor_slices(train_x).batch(args.batch_size)
    eval_dataset =  batch_data(val_x, args)     # tf.data.Dataset.from_tensor_slices(val_x).batch(args.batch_size)
    test_dataset =  batch_data(test_x, args)    # tf.data.Dataset.from_tensor_slices(test_x).batch(args.batch_size)

    adam_optimizer = optimizer(args.lr)

    # History of the learning and evaluation
    # processes trough epochs
    train_history = {'loss':[], 'RE':[], 'KL':[]}
    eval_history = {'loss':[], 'RE':[], 'KL':[]}

    # Time history
    time_history = list()

    current_epoch = 0
    best_loss = 1e10
    experiment_begin_time = time.time()
    while (current_epoch < args.epochs):

        current_epoch += 1
        tf.print("----- EPOCH {}/{} -----".format(current_epoch,args.epochs))

        epoch_start_time = time.time()
        train_loss_epoch, train_RE_epoch, train_KL_epoch = one_pass(
            model,
            train_dataset,
            mode = 'train',
            optimizer = adam_optimizer
        )

        eval_loss_epoch, eval_RE_epoch, eval_KL_epoch = one_pass(
            model,
            eval_dataset,
        )

        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        experiment_elapsed_time = time.time() - experiment_begin_time

        # Breaking if the loss increased
        if (eval_loss_epoch <= best_loss):
            best_loss = eval_loss_epoch
        # else:
        #     break

        # if the loss increased we don't add this epoch to the history
        # update the process history
        train_history['loss'].append(train_loss_epoch)
        train_history['RE'].append(train_RE_epoch)
        train_history['KL'].append(train_KL_epoch)

        eval_history['loss'].append(eval_loss_epoch)
        eval_history['RE'].append(eval_RE_epoch)
        eval_history['KL'].append(eval_KL_epoch)

        time_history.append(experiment_elapsed_time)

        # printing results
        tf.print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
                '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
                'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
                '\n'.format(
            current_epoch, args.epochs, epoch_elapsed_time,
            train_loss_epoch, train_RE_epoch, train_KL_epoch,
            eval_loss_epoch, eval_RE_epoch, eval_KL_epoch,
        ))

    tf.print("Plotting history")
    plot_history(train_history, eval_history, time_history, args)

    tf.print("Saving weights")
    model.save_weights(args.job_dir + "model_weights.save")

    # Out the while loop
    # At this point we have the best model
    # We test it now on the test dataset
    tf.print("Calculating test loss")
    test_loss, test_KL, test_RE = one_pass(model, test_dataset)
    
    tf.print("Calling evaluate_model()")
    log_likelihood_test, log_likelihood_train, elbo_test, elbo_train = evaluate_model(model, train_x, test_x, args)

    # Print the results of the test
    with file_io.FileIO(args.job_dir + 'final_results.txt', 'w') as f:
        print('FINAL EVALUATION ON TEST SET\n'
              'LogL (TEST): {:.2f}\n'
              'LogL (TRAIN): {:.2f}\n'
              'ELBO (TEST): {:.2f}\n'
              'ELBO (TRAIN): {:.2f}\n'
              'Loss: {:.2f}\n'
              'RE: {:.2f}\n'
              'KL: {:.2f}'.format(
            log_likelihood_test,
            log_likelihood_train,
            elbo_test,
            elbo_train,
            test_loss,
            test_RE,
            test_KL
        ), file=f)
