import tensorflow as tf
from tensorflow import keras

@tf.function
def train_step(model, optimizer, loss):
    """ Performs a training step for a set of samples """

    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Here we train the model on the train dataset
def one_pass(model, dataset, mode = 'evaluate', optimizer = None):
    """
        One pass through the model (1 epoch)
        Can be for training or just for evaluation
    """

    RE = 0
    loss = 0
    KL = 0

    for batch_data in dataset:
        loss_stamp, RE_stamp, KL_stamp = model.loss(batch_data)
        
        if mode == 'train':
            train_step(model, optimizer, loss_stamp)
            
        RE += RE_stamp
        loss += loss_stamp
        KL += KL_stamp

    # We average loss, KL and RE over batch size
    RE /= len(dataset)
    loss /= len(dataset)
    KL /= len(dataset)

    return loss, KL, RE