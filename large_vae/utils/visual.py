import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_images(args, x_sample, dir, file_name, size_x=4, size_y=4):

    fig = plt.figure(figsize=(size_x, size_y))

    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

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

    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)

def plot(data, time, train_or_eval, type, dir):

    fig_name = '{} {} history'.format(
        train_or_eval, 
        type
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('time')
    ax.set_ylabel(type)
    ax.set_title(fig_name)
    plt.plot(time, data)
    plt.savefig(dir + fig_name + '.png', bbox_inches='tight')


# TODO: Plot the history
def plot_history(train_history, eval_history, time_history, dir):

    # Plotting train history
    for metric_name, metric_data in train_history.items(): 
        plot(metric_data, time_history, 'train', metric_name, dir)

    for metric_name, metric_data in eval_history.items():
        plot(metric_data, time_history, 'evaluation', metric_name, dir)

