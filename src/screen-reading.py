import time

import cv2
import numpy as np
from PIL import ImageGrab

from src.directkeys import PressKey, W, A, S, D


def screen_record():
    last_time = time.time()
    while (True):
        printscreen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        print('loop took {} secs'.format(time.time() - last_time))
        last_time = time.time()
##        cv2.imshow('window', cv32.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break
def process_img (image):
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img,threshold1=200, threshold2=300)
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]])

    processed_img = roi(processed_img,[vertices])

    lines = cv2.HoughLines(processed_img,)
    return processed_img


def roi(img, vertices):
    # blank mask
    mask = np.zeros_like(img)
    # fill the mask 
    cv2.fillPoly(mask,vertices,255)
    # now only show the area that is the mask 
    masked = cv2.bitwise_and(img,mask)
    return masked 


def main():
    for i in range(4)[::-1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()
    while True: 
##        PressKey(W)
        screen = np.array(ImageGrab.grab(bbox= (0,40,800,600)))
        last_time = time.time()
        new_screen = process_img(screen)
        cv2.imshow('window',new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
