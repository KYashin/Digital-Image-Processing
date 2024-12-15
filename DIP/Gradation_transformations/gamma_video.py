import cv2 as cv
import numpy
import numpy as np


def gamma_correction(gamma):
    cap = cv.VideoCapture(0)

    while True:
        success, frame = cap.read()

        frame_1 = np.clip(255 * (frame / 255) ** gamma, 0, 255).astype(np.uint8)

        cv.imshow("Frame", frame)
        cv.imshow("Frame_1", frame_1)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()

gamma_correction(0.2)
