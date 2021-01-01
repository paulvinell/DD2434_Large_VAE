#
# Notes:
# Created using Tensorflow 2.4.0
#
from utils.parse_arg import parse_arguments
from utils.load_data import load_experiment_dataset
from utils.load_model import load_model, create_experiment_folder
from experiment.runexperiment import run_experiment

def main():
    # Parse arguments
    args = parse_arguments()

    # Load unbatched experiment dataset
    train_dataset, eval_dataset, test_dataset = load_experiment_dataset(args)

    # Initialize the model
    vae = load_model(args)
    
    # Directory where to save the results of this experiment
    experiment_folder = create_experiment_folder(args, __file__)

    # Run the experiment
    run_experiment(vae, train_dataset, eval_dataset, test_dataset, experiment_folder, args)


if __name__ == "__main__":
    main()
