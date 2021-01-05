import os
import numpy as np
from tensorflow.python.lib.io import file_io

def save_data(data, name, folder_name):
    """ Save all the passed data generated 
        as a fille
    """ 
    filepath = os.path.join(folder_name, name + ".csv")
    with file_io.FileIO(filepath, 'w') as f:
        np.savetxt(f, data)
        f.close()
    


