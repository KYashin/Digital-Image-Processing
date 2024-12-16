import numpy as np
import cv2

# Загрузим изображение и преобразуем в grayscale
I = cv2.imread(r"C:\Users\user\PycharmProjects\Digital-Image-Processing\DIP\Images_DIP\Lenna_test_image.png", cv2.IMREAD_GRAYSCALE) / 255.0
p = I.copy()

def window_sum(matrix, r):
    """
    Вычисляет сумму значений в окне радиуса r для каждого пикселя изображения.
    """
    height, width = matrix.shape
    result = np.zeros_like(matrix, dtype=np.float64)

    for y in range(height):
        for x in range(width):
            # Определяем границы окна
            y1, y2 = max(0, y - r), min(height, y + r + 1)
            x1, x2 = max(0, x - r), min(width, x + r + 1)
            result[y, x] = np.sum(matrix[y1:y2, x1:x2])

    return result

def guided_filter_manual(I, p, r, epsilon):
    """
    Реализует Guided Image Filter без встроенных функций.

    :param I: Направляющее изображение (I)
    :param p: Входное изображение (p)
    :param r: Радиус окна
    :param epsilon: Параметр регуляризации
    :return: Отфильтрованное изображение
    """

    # Размер изображения
    height, width = I.shape
    N = window_sum(np.ones((height, width)), r)  # Количество пикселей в каждом окне

    # 1. Средние значения для I и p
    mean_I = window_sum(I, r) / N
    mean_p = window_sum(p, r) / N

    # 2. Средние значения для I*p и I^2
    mean_Ip = window_sum(I * p, r) / N
    mean_II = window_sum(I * I, r) / N

    # 3. Ковариация cov_Ip и дисперсия var_I
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # 4. Вычисление коэффициентов a и b
    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I

    # 5. Усреднение коэффициентов a и b
    mean_a = window_sum(a, r) / N
    mean_b = window_sum(b, r) / N

    # 6. Получаем финальное изображение q
    q = mean_a * I + mean_b
    return q

r = 100  # Радиус окна
epsilon = 1e-3  # Параметр регуляризации

q = guided_filter_manual(I, p, r, epsilon)

cv2.imshow("Image", I)
cv2.imshow("Filtered Image", (q * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()