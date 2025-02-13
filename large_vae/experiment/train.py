import tensorflow as tf
from tensorflow import keras
import time

@tf.function
def train_step(model, optimizer, x):
    """ Performs a training step for a set of samples """

    with tf.GradientTape() as tape:
        loss, RE, KL = model.loss(x, average=True)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, RE, KL


# Here we train the model on the train dataset
def one_pass(model, dataset, mode = 'evaluate', optimizer = None):
    """
        One pass through the model (1 epoch)
        Can be for training or just for evaluation
    """

    RE = 0
    loss = 0
    KL = 0

    used_data = 0
    total_data = len(dataset)
    batch_time = time.time()

    for batch_data in dataset:
        if mode == 'train':
            loss_stamp, RE_stamp, KL_stamp = train_step(model, optimizer, batch_data)
        elif mode == 'evaluate':
            loss_stamp, RE_stamp, KL_stamp = model.loss(batch_data, average = True)

        RE += RE_stamp
        loss += loss_stamp
        KL += KL_stamp

        used_data += len(batch_data)

        # tf.print("Processed {}/{} (batch took {}s)".format(used_data, batch_data.shape[0] * total_data, time.time() - batch_time))
        batch_time = time.time()

    # We average loss, KL and RE over batch size
    RE /= len(dataset)
    loss /= len(dataset)
    KL /= len(dataset)

    return loss, KL, RE
