import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("D:\pythonProject\Images\i.jpg", cv.IMREAD_GRAYSCALE)

histogram = [0] * 256                # действие вложенного цикла аналогично тому, что делает функция np.bincount
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        intensity = img[y, x]
        histogram[intensity] += 1
