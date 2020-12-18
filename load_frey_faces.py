# Thanks to Elvis Dohmatob for the original loading script of Frey Faces
# His blog post: https://dohmatob.github.io/research/2016/10/22/VAE.html
# Differences: I changed from Python 2 -> Python 3

import os
from urllib.request import urlopen, URLError, HTTPError
from scipy.io import loadmat

def fetch_file(url):
    try:
        f = urlopen(url)
        print("Downloading data file " + url + " ...")

        # Open our local file for writing
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())
        print("Done.")

    #handle errors
    except HTTPError as e:
        print("HTTP Error:", e.code, url)
    except URLError as e:
        print("URL Error:", e.reason, url)

def load_frey_faces():
    url =  "http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat"
    data_filename = os.path.basename(url)
    if not os.path.exists(data_filename):
        fetch_file(url)
    else:
        print("Data file %s exists." % data_filename)

    # reshape data for later convenience
    img_rows, img_cols = 28, 20
    ff = loadmat(data_filename, squeeze_me=True, struct_as_record=False)
    ff = ff["ff"].T.reshape((-1, img_rows, img_cols))

    return ff
