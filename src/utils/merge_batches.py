from utils import constants
import numpy as np


def extract_digit(string):
    import re
    return int(re.findall("\d+", string)[0])


def load_all_batches():
    with open(constants.CSV_BATCH_RECORD, "r") as records:
        rows = records.read().strip().split("\n")
        file_names = list(set(map(lambda x: x.split(",")[1], rows)))

        data = list()
        for file_name in sorted(file_names, key=lambda x: extract_digit(x)):
            print("Loading data from" + file_name)
            data.extend(np.load(file_name))
        return data


if __name__ == '__main__':
    np.save(constants.TRAINING_DATA_PATH, load_all_batches())
