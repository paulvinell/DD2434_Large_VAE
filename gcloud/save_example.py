from tensorflow.python.lib.io import file_io

something = ['a', 'b', 'c', 1, 2, 3]

# Saves to gcloud storage 'large_vae', subfolder results, file something.pickle
with file_io.FileIO('gs://large_vae/results/something.pickle', 'wb') as something:
    pickle.dump(something, f)
