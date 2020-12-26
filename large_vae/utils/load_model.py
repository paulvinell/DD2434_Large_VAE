from models.vae import VAE

def load_model(args):

    if args.model == 'vae':
        model = VAE(args)
    elif args.model == 'hvae':
        pass
    return model