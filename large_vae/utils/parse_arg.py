import argparse

######### All possible arguments #########
#+- Training +-#
# epochs:
#   - number of epochs to use during the training
#   - possible values = positive number
# batch_size:
#   - Batches size
#   - possible values = positive number
# lr:
#   - learning rate of the learning process
#   - possible values = at most 1
# early_stopping:
#   - Number of epochs for early stopping
#   - possible values = at most epoch
# warmup:
#   - number of epoch for warmup
#   - possible values = positive number


#+- Model +-#
# model:
#   - type of model to use
#   - possible values = 'vae', 'hvae'
# prior:
#   - type of prior to use
#   - possible values = 'vamp', 'mog', 'gaussian'
# pseudoinputs:
#   - number of pseudoinputs of the vampprior
#   - possible values = positive number
# z1_size:
#   - First latent layer size
#   - possible values = positive number
# z2_size
#   - second latent layer size
#   - possible values = positive number
#TODO: Maybe add some additional pseudoinputs-specific arguments


#+- Dataset +-#
# dataset_name:
#   - name of the dataset
#   - possible values = 'minst', 'frey'
########################################################################

# utility class to use in argparse
class interval:
    """Check if the string corresponds
        to a integer between mini and maxi
    """
    def __init__(self, mini=0, maxi=float('inf')):
        self.mini = mini
        self.maxi = maxi

    def check(self, x):
        try:
            int_ = int(x)
            if int_ < self.mini or int_ > self.maxi:
                raise argparse.ArgumentTypeError(
                    "should be a number between {} and {}.".format(self.mini,self.maxi)
                )
            return int_
        except ValueError:
            raise argparse.ArgumentTypeError(
                "should be a number between {} and {}.".format(self.mini,self.maxi))

# Parsing all possible arguments
# They are all described in the comments above
def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run a VAE experiment on a dataset"
    )

    pos_num = interval().check
    below_one = interval(maxi = 1).check

    #+- Training +-#
    parser.add_argument("--epochs", type=pos_num,default=100, help="number of epochs to use during the training")
    parser.add_argument("--batch_size", type=pos_num,default=16, help="batch size to use for training")
    parser.add_argument("--lr", type=below_one, default=0.0005,help="learning rate of the learning process" )
    parser.add_argument("--evaluation", type=below_one, default=0.1,help="percentage of training data to be used for evaluation" )
    parser.add_argument("--early_stopping", type=pos_num,default=50, help="number of epochs for early stopping")
    parser.add_argument("--warmup", type=pos_num,default=100, help="number of epochs for warmup")

    #+- Model +-#
    parser.add_argument("--model", type=str, default='vae', choices=['vae', 'hvae'], help="type of model to use")
    parser.add_argument("--prior", type=str, default='gaussian', choices=['gaussian', 'vampprior', 'mog'], help = "prior to use")

    parser.add_argument("--z1_size", type=pos_num, default=40, help = "first latent layer size")
    parser.add_argument("--z2_size", type=pos_num, default=40, help = "second latent layer size")
    parser.add_argument("--dataset", type=str, default='mnist', choices=['mninst', 'frey'], help = "dataset name")
    parser.add_argument("--pseudoinput_count", type=int, default=500, help='number of pseudoinputs for the VampPrior')

    args = parser.parse_args()

    args_vars = vars(args)
    if args.dataset == 'mnist':
        args_vars['input_size'] = (28, 28, 1)
    elif args.dataset == 'frey':
        args_vars['input_size'] = (20, 28, 1)

    return args
