# -*- coding: utf-8 -*-
"""
Downloads a demo image

@author: Fredrik Wahlberg <fredrik.wahlberg@it.uu.se>
"""

fn = 'manuscript.jpg'
url = """https://upload.wikimedia.org/wikipedia/commons/4/46/Calligraphy.malmesbury.bible.arp.jpg"""


def getColourImage():
    import os.path
    if not os.path.exists(fn):
        import urllib.request
        urllib.request.urlretrieve(url, fn)
    import cv2
    return cv2.imread(fn)


def getGrayScaleImage():
    import cv2
    return cv2.cvtColor(getColourImage(), cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(getColourImage())
    plt.subplot(1, 2, 2)
    plt.imshow(getGrayScaleImage(), cmap='gray')
    plt.show()
