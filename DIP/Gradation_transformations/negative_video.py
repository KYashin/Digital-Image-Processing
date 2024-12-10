import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    frame_1 = frame.astype(np.int32)

    frame_1 = 255 - frame

    frame_1 = frame_1.astype(np.uint8)

    cv2.imshow("Frame_1", frame_1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()