import time

import cv2
import numpy as np
from utils.constants import *
from nn.alexnet import alexnet
from keys.directkeys import PressKey, ReleaseKey, W, A, D, S
from keys.get_keys import key_check
from capture_screen import grab_screen
from wait import wait

WIDTH = 160
HEIGHT = 120
LR = 1e-3
N_EPOCH = 8

MODEL_NAME = "PythonGTA5_{}_{}_{}_Epochs.model".format(LR, "alexnet", N_EPOCH)
MODEL_NAME = "PythonGTA5_SUPERBIKE_{}_{}_{}_Epochs.model".format(LR, "alexnet", N_EPOCH)
KEY_PRESS_TIME = 0.09


def forward():
    print("FORWARD")
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(W)
    time.sleep(KEY_PRESS_TIME)
    PressKey(S)
    time.sleep(0.05)
    ReleaseKey(S)


def left():
    print("LEFT")
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    time.sleep(KEY_PRESS_TIME)


def right():
    print("RIGHT")
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    time.sleep(KEY_PRESS_TIME)


def release_all():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def main(model):
    last_time = time.time()

    paused = False
    while True:
        if not paused:
            print('Frame took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
            screen = cv2.cvtColor(grab_screen(region=(0, 40, 800, 600)), cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            moves = list(np.around(model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]))
            print(moves)

            if moves == [1, 0, 0] or moves == [1, 1, 0]:
                left()
            elif moves == [0, 1, 0]:
                forward()
            elif moves == [0, 0, 1]:
                right()

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                print("Resuming...")
                time.sleep(1)
                print("Resumed")
                release_all()

            else:
                print("Pausing...")
                paused = True
                time.sleep(1)
                print("Paused")
                release_all()
        if 'G' in keys:
            release_all()
            break


if __name__ == '__main__':
    model = alexnet(WIDTH, HEIGHT, LR)
    model_path = MODEL_DIR + "/" + MODEL_NAME
    model.load(model_path)
    wait(4)
    print(model_path)
    main(model=model)
