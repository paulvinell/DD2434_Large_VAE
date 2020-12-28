import time

import tensorflow as tf

from experiment.train import one_pass
from experiment.evaluate import evaluate_model

def optimizer(lr):
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    return optimizer

# TODO: Plot the history
def plot_history(train_history, eval_history, time_history):
    pass


def run_experiment(model, train_dataset, val_dataset, eval_dataset, args):

    """ Run the complete experiment.
        That is, for each epoch, train the dataset and
        evaluate it.

        The model is ready when the loss function has increased
        over two successive epochs or when we reach the max number
        of epochs passed as argument in CLI.

    """

    adam_optimizer = optimizer(args.lr)

    # History of the learning and evaluation
    # processes trough epochs
    train_history = {'loss':[], 'RE':[], 'KL':[]}
    eval_history = {'loss':[], 'RE':[], 'KL':[]}

    # Time history
    time_history = list()

    current_epoch = 0
    best_loss = 1e10

    while (current_epoch <= args.epochs):

        current_epoch += 1

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

        epoch_elapsed_time = time.time() - epoch_start_time

        # Breaking if the loss increased
        if (eval_loss_epoch <= best_loss):
            best_loss = eval_loss_epoch
        else:
            break

        # if the loss increased we don't add this epoch to the history
        # update the process history
        train_history['loss'].append(train_loss_epoch)
        train_history['RE'].append(train_RE_epoch)
        train_history['KL'].append(train_KL_epoch)

        eval_history['loss'].append(eval_loss_epoch)
        eval_history['RE'].append(eval_RE_epoch)
        eval_history['KL'].append(eval_KL_epoch)

        time_history.append(epoch_elapsed_time)

        # printing results
        print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
                '* Train loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
                'o Val.  loss: {:.2f}   (RE: {:.2f}, KL: {:.2f})\n'
                '\n'.format(
            current_epoch, args.epochs, epoch_elapsed_time,
            train_loss_epoch, train_RE_epoch, train_KL_epoch,
            eval_loss_epoch, eval_RE_epoch, eval_KL_epoch,
        ))

    # TODO: It may be interesting to plot the process history. Complete this function
    plot_history(train_history, eval_history, time_history)


    # Out the while loop
    # At this point we have the best model
    # We evaluate it now
    # TODO: We need a training, evaluation and test dataset we only have the training and test ones. Can be changed in the load dataset module
    # evaluate_model(model, test_dataset, args)
