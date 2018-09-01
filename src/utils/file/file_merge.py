import os

import numpy as np


def extract_digit(string):
    import re
    return int(re.findall("\d+", string)[0])


def merge_npy_files(path):
    data = list()

    for data_path in sorted(os.listdir(path), key=lambda x: extract_digit(x)):
        print("Loading data from file {}".format(data_path))
        data.extend(np.load(path+"/"+data_path))
    return data

