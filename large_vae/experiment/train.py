
# Here we train the model omn the train dataset
def train_model(model, args, train_dataset):

    input("Press Enter to begin...")
    for epoch in range(1, 1 + args.epochs):
        print("Epoch {}".format(epoch))
        for x in train_dataset:
            model.train_step(x)