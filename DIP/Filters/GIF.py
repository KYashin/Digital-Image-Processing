import numpy as np
import scipy.ndimage as ndimage
import cv2 as cv
import matplotlib.pyplot as plt

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

    # 1. Вычисляем среднее значение для изображений I и p в окне радиуса r
    mean_I = ndimage.uniform_filter(I, size=2 * r + 1)
    mean_p = ndimage.uniform_filter(p, size=2 * r + 1)

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

image = cv.imread(r'C:\Users\user\PycharmProjects\Digital-Image-Processing\DIP\Images_DIP\Lenna_test_image.png', cv.IMREAD_GRAYSCALE)

# Параметры
r = 150  # Радиус окна
epsilon = 0.9  # Параметр регуляризации

noise = np.random.normal(0, 1000, image.shape).astype(np.float32)
noisy_image = image.astype(np.float32) + noise
noisy_image = np.clip(noisy_image, 0, 255)

# Применяем фильтр
output = (guided_filter(image.astype(np.float32), noisy_image, r, epsilon)).astype(np.uint8)

# cv.imshow("Input image", I)
# cv.imshow("Noisy image", noisy_image)
# cv.imshow("Filtered image", output)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Визуализация
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Filtered Image")
plt.imshow(output, cmap="gray")
plt.axis("off")

plt.show()

