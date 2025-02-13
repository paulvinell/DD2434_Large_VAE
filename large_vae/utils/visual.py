import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorflow.python.lib.io import file_io

from large_vae.utils.data_saving import save_data

def plot_images(x_sample, file_name, args, size_x=4, size_y=4):

    fig = plt.figure(figsize=(size_x, size_y))

    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    #Save the data in a file before plotting it
    save_data(x_sample, file_name, args.job_dir)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape((args.input_size[2], args.input_size[1], args.input_size[0]))
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)

        # if args.input_type == 'gray':
        #     sample = sample[:, :, 0]
        #     plt.imshow(sample, cmap='gray')
        # else:
        plt.imshow(sample)

    plt_save(args, file_name + '.png')
    plt.close(fig)

def plot(args, data, time, train_or_eval, type):
    fig_name = '{} {} history'.format(
        train_or_eval,
        type
    )
    #Save the data in a file before plotting it
    save_data(np.array(data), fig_name, args.job_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('epochs')
    ax.set_ylabel(type)
    ax.set_title(fig_name)
    plt.plot(time, data)
    plt_save(args, fig_name + '.png')

def plot_history(train_history, eval_history, time_history, args):

    #Save time history in a file
    save_data(time_history, 'time', args.job_dir)

    # Plotting train history
    for metric_name, metric_data in train_history.items():
        plot(args, metric_data, time_history, 'train', metric_name)

    for metric_name, metric_data in eval_history.items():
        plot(args, metric_data, time_history, 'evaluation', metric_name)


# Do you think this is a terribly complicated way to save a plot?
# It is.
def plt_save(args, name):
    filepath = args.job_dir + name
    with file_io.FileIO(filepath, 'wb') as file:
        plt.savefig(file, bbox_inches='tight')
