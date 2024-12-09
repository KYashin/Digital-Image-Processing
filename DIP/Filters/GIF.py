import numpy as np
import cv2

def guided_filter_with_weights(I, P, r, eps):
    """
    Guided Filter с использованием весовой функции.
    :param I: направляющее изображение (градации серого, float32)
    :param P: обрабатываемое изображение (градации серого, float32)
    :param r: радиус окна
    :param eps: регуляризующий параметр
    :return: отфильтрованное изображение
    """
    # Размер окна
    kernel_size = 2 * r + 1
    window_area = kernel_size ** 2

    # Локальное среднее
    mean_I = cv2.boxFilter(I, -1, (kernel_size, kernel_size))
    mean_P = cv2.boxFilter(P, -1, (kernel_size, kernel_size))

    # Локальная дисперсия
    mean_I2 = cv2.boxFilter(I * I, -1, (kernel_size, kernel_size))
    var_I = mean_I2 - mean_I ** 2

    # Ковариация
    mean_IP = cv2.boxFilter(I * P, -1, (kernel_size, kernel_size))
    cov_IP = mean_IP - mean_I * mean_P

    # Весовая функция
    weights = (1 + (I - mean_I) * (P - mean_P) / (var_I + eps)) / window_area

    # Фильтрованное изображение
    q = cv2.boxFilter(weights * P, -1, (kernel_size, kernel_size))
    return q

# Пример использования
image = cv2.imread(r"C:\Users\user\PycharmProjects\Digital-Image-Processing\DIP\Images_DIP\Lenna_test_image.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
filtered = guided_filter_with_weights(image, image, r=8, eps=0.02) * 255

# Нормализуем результат, чтобы вывести его
filtered = (filtered * 255).astype(np.uint8)

cv2.imshow("Original", image)
cv2.imshow("Filtered", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

