import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    frame_1 = frame.astype(np.int32)

    frame_2 = frame.astype(np.int32)

    frame_1 += 100

    frame_2 -= 100

    frame_1 = np.clip(frame_1, 0, 255)

    frame_2 = np.clip(frame_2, 0, 255)

    frame_1 = frame_1.astype(np.uint8)

    frame_2 = frame_2.astype(np.uint8)

    cv2.imshow("Frame_1", frame_1)
    cv2.imshow("Frame", frame)
    cv2.imshow("Frame_2", frame_2)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()