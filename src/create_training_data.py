import os
import time
from datetime import datetime

import cv2
import numpy as np

from utils.constants import *
from keys.get_keys import key_check
from capture_screen import grab_screen
import csv

from wait import wait


def keys_to_output(keys):
    """
    convert keys to a muli array
    [A,W,D] boolean values
    :param keys:
    :return: output
    """
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1

    if 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


def get_latest_batch():
    with open(CSV_BATCH_RECORD, "r") as batch_record:
        recent_batch = batch_record.read().strip().split("\n")[-1]
        batch = recent_batch.split(",")
        if batch[0] == "":
            return 1
        if int(batch[2]) % 1000 == 0:
            return int(batch[0]) + 1
        else:
            return int(batch[0])


def save_batch_info(row):
    with open(CSV_BATCH_RECORD, "a", newline='') as csv_file:
        batch_record = csv.writer(csv_file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        batch_record.writerow(row)


def save_training_batch(training_data, batch_number):
    num_batches = str(batch_number)
    data_size = str(len(training_data))
    if batch_number == 1:
        print("Creating a fresh batch")
        file_name = TRAINING_BATCH.format(1)
        print(str(datetime.now()) + ": Size :" + str(len(training_data)))
        print("Saving to {}".format(file_name))
        np.save(file_name, training_data)
        save_batch_info([batch_number, file_name, data_size])
    else:
        file_name = TRAINING_BATCH.format(num_batches)
        print(str(datetime.now()) + ": Size :" + str(len(training_data)))
        print("Saving to {}".format(file_name))
        np.save(file_name, training_data)
        save_batch_info([batch_number, file_name, data_size])


# print(str(datetime.now()) + ": Size :" + str(len(training_data)))
# print("Saving to {}".format(file_name))


def train_ai():
    batch_number = get_latest_batch()
    file_name = TRAINING_BATCH.format(batch_number)

    if os.path.isfile(file_name):
        print('File {} already exists, loading previous data'.format(file_name))
        training_data = list(np.load(file_name))
        print(len(training_data))
    else:
        print('Creating a new file ' + file_name)
        training_data = list()

    paused = False
    while True:
        if not paused:
            screen = grab_screen(region=(0, 40, 800, 600))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160, 120))
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, output])
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     print("Q pressed exiting")
            #     cv2.destroyAllWindows()
            #     break
            if (len(training_data)) % 1000 == 0:
                save_training_batch(training_data, batch_number)
                training_data = list()
                batch_number += 1

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                print("Resuming...")
                time.sleep(1)
                print("")
            else:
                print("Pausing...")
                paused = True
                time.sleep(1)

        if "G" in keys:
            save_training_batch(training_data, batch_number)
            break


if __name__ == "__main__":
    wait(4)
    train_ai()
    print(get_latest_batch())
    # file_name = TRAINING_DATA_PATH
    #
    # if os.path.isfile(file_name):
    #     print('File already exists, loading previous data')
    #     training_data = list(np.load(file_name))
    #     print(len(training_data))
    # else:
    #     print('Creating a new file ' + file_name)
    #     training_data = list()
    # batch_number = get_latest_batch()
    # for i in range(5):
    #     save_training_batch(training_data[:1000], batch_number)
    #     batch_number += 1
    # save_training_batch(training_data[1000:])
