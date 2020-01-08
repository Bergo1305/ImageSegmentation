import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram(image_path, channel=None, mask=None, hist_size=None, ranges=None) -> None:
    if channel is None:
        channel = [0]
        img = cv2.imread(image_path, 0)
    else:
        img = cv2.imread(image_path)

    if hist_size is None:
        hist_size = [256]

    if ranges is None:
        ranges = [0, 256]

    if mask is not None:
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[100:300, 100:400] = 255
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        hist = cv2.calcHist([img], channel,  masked_img, hist_size, ranges)
    else:
        hist = cv2.calcHist([img], channel, None, hist_size, ranges)

    plt.plot(hist)
    plt.show()
