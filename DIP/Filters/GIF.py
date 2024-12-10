import numpy as np
import scipy.ndimage as ndimage
import cv2 as cv

def guided_filter(I, p, r, epsilon):
    """
    Реализует Guided Image Filter.

    :param I: Направляющее изображение (I)
    :param p: Входное изображение (p)
    :param r: Радиус окна
    :param epsilon: Параметр регуляризации
    :return: Отфильтрованное изображение
    """

    # Размер изображения
    height, width = I.shape
    print("Height:", height)
    print("Width:", width)

    # 1. Вычисляем среднее значение для изображений I и p в окне радиуса r
    mean_I = ndimage.uniform_filter(I, size=2 * r + 1)
    mean_p = ndimage.uniform_filter(p, size=2 * r + 1)

    print(mean_I.shape)
    print(mean_I)

    # 2. Вычисляем дисперсию для изображения p
    var_I = ndimage.uniform_filter(I ** 2, size=2 * r + 1) - mean_I ** 2
    # var_p = ndimage.uniform_filter(p ** 2, size=2 * r + 1) - mean_p ** 2
    cov_Ip = ndimage.uniform_filter(I * p, size=2 * r + 1) - mean_I * mean_p

    # 3. Вычисляем коэффициент a (линейный коэффициент)
    a = cov_Ip / (var_I + epsilon)

    # 4. Вычисляем смещение b
    b = mean_p - a * mean_I

    # 5. Вычисляем финальное отфильтрованное изображение
    mean_a = ndimage.uniform_filter(a, size=2 * r + 1)
    mean_b = ndimage.uniform_filter(b, size=2 * r + 1)

    q = mean_a * I + mean_b

    return q

I = cv.imread(r'D:\pythonProject\DIP\Images_DIP\tmb_120917_5876.jpg', cv.IMREAD_GRAYSCALE).astype(np.float32) / 255

# Параметры
r = 4  # Радиус окна
epsilon = 0.04  # Параметр регуляризации

# Применяем фильтр
output = (guided_filter(I, I, r, epsilon) * 255).astype(np.uint8)
output_1 = output.astype(np.uint8)

cv.imshow("Input image", I)
cv.imshow("Filtered image", output)
cv.waitKey(0)
cv.destroyAllWindows()

