def preprocess(data):
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1)) / 255.0
    return data.astype('float32')
