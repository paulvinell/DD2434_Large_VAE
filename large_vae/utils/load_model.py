import os
import json
import errno

from large_vae.models.vae import VAE

def load_model(args):

    if args.model == 'vae':
        model = VAE(args)
    elif args.model == 'hvae':
        pass
    return model


def create_experiment_folder(args, project_folder):
    """ Creates the folder where to save the results of the experiment
        The name of the folder is generated automatically from the
        parameters of the experiment (args).

        It also saves the experiment parameters as a json file in the
        resulting folder.

    """

    # Generating the folder absolute path
    dict_args = vars(args).copy()

    # Don't include unnecessary identifiers
    dict_args.pop('input_size')
    dict_args.pop('job_dir')
    dict_args.pop('gcloud')
    if dict_args.get('model') == 'vae':
        dict_args.pop('z2_size')

    expe_folder_abs_path = os.path.join(
        project_folder,
        "results",
        dict_args.pop('model'),
        dict_args.pop('prior'),
        dict_args.pop('dataset'),
        str(dict_args).replace('\'', '').replace('{', '').replace('}', '').replace(': ', '-'),
        ""
    )

    # Creating the folder
    if not os.path.exists(expe_folder_abs_path):
        try:
            os.makedirs(expe_folder_abs_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # Save the json file containing the parameters in the folder
    param_path = os.path.join(expe_folder_abs_path, 'parameters.json')
    with open(param_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
        f.close()

    return expe_folder_abs_path
