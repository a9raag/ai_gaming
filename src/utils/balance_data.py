from collections import Counter
from random import shuffle

import cv2
import numpy as np
import pandas as pd

from utils import constants


def view_img_data(img_choice_data):
    for data in img_choice_data:
        img, choice = data[0], data[1]
        cv2.imshow('test', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    print("Loading data")
    train_data = np.load(constants.TRAINING_DATA_PATH)
    print("Data Loaded of size {}".format(len(train_data)))
    df = pd.DataFrame(train_data)
    print(df.head(10))
    print(Counter(df[1].apply(str)))

    lefts, rights, forwards = list(), list(), list()

    for data in train_data:
        img, choice = data[0], data[1]

        if choice == [1, 1, 0]:
            lefts.append(data)

        elif choice == [0, 1, 0]:
            forwards.append(data)

        elif choice == [0, 0, 1]:
            rights.append(data)

            # else:
            # print("None of the actions matched!!")

    print("Forwards ", len(forwards))
    print("Lefts ", len(lefts))
    print("Rights ", len(rights))
    forwards = forwards[:len(lefts)][:len(rights)]
    lefts = lefts[:len(forwards)]
    rights = rights[:len(forwards)]

    final_data = forwards + lefts + rights
    shuffle(final_data)
    np.save(constants.FINAL_TRAINING_DATA, final_data)
    print("Saved final data")