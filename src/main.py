import time

import cv2

from capture_screen import grab_screen
from vision.process_img import process_img
from keys.directkeys import PressKey, ReleaseKey, W, A, D
from wait import wait
from darkflow.net.build import TFNet
from detect_objects import detect_objects


# def screen_record():
#     last_time = time.time()
#     while (True):
#         printscreen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
#         print('loop took {} secs'.format(time.time() - last_time))
#         last_time = time.time()
#         # cv2.imshow('window', cv32.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
#         if (cv2.waitKey(25) & 0xFF == ord('q')):
#             cv2.destroyAllWindows()
#             break
#
#


def straight():
    print("Forward")
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    print("LEFT")
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    ReleaseKey(W)


def right():
    print("RIGHT")
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)


def slow_down():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


options = {"model": "D:/darkflow/cfg/yolov2-tiny.cfg", "load": "D:/darkflow/bin/yolov2-tiny.weights",
           "threshold": 0.1,
           "labels": "D:/darkflow/labels.txt",
           "gpu": 0.3
           }

tfnet = TFNet(options)


def self_drive():
    last_time = time.time()

    while True:
        screen = grab_screen(region=(0, 40, 800, 640))
        print('Frame took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        new_screen, original_image, m1, m2 = process_img(screen)
        object_predictions = detect_objects(tfnet, screen)
        # cv2.imshow('window', new_screen)q
        # cv2.imshow('window2', cv2.cvtColor(object_predictions, cv2.COLOR_BGR2RGB))
        cv2.imshow('window2', object_predictions)

        # if m1 < 0 and m2 < 0:
        #     right()
        # elif m1 > 0 and m2 > 0:
        #     left()
        # else:
        #     straight()

        # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    wait(4)
    self_drive()
