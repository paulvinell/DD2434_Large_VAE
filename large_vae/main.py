#
# Notes:
# Created using Tensorflow 2.4.0
#
from utils.parse_arg import parse_arguments
from utils.load_data import load_experiment_dataset
from utils.load_model import load_model
from experiment.runexperiment import run_experiment

# Parse arguments 
args = parse_arguments()

# Load experiment dataset
train_dataset, test_dataset = load_experiment_dataset(args.dataset)

# Initialize the model
vae = load_model(args)

# Run the experiment
run_experiment(vae, train_dataset, test_dataset, args)