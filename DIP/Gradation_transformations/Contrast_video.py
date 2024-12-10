import cv2 as cv
import numpy
import numpy as np


def contrast_video(coefficient):
    cap = cv.VideoCapture(0)

    while True:
        success, frame = cap.read()
        frame_1 = frame.astype(np.float32)
        frame_1 *= coefficient

        frame_1 = np.clip(frame_1, 0, 255).astype(np.uint8)

        cv.imshow("Frame", frame)
        cv.imshow("Frame_1", frame_1)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()

contrast_video(2.5)
