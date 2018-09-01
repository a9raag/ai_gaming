# getkeys.py
# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

keyList = ["\b", chr(27)]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


