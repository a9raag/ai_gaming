import time


def wait(n=4):
    for i in range(n, 1, -1):
        print(i)
        time.sleep(1)