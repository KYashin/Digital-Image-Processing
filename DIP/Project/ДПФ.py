import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Загрузка изображения
img = cv.imread(r"D:\pythonProject\DIP\Project\Images\Lenna_test_image.png", cv.IMREAD_GRAYSCALE)

# Приведение изображения к типу np.float32 для обработки
img = np.float32(img)

# Приведение изображения к комплексному типу для работы с БПФ
img = np.complex64(img)

# Функция для нахождения ближайшей степени двойки
def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

# Приведение размеров изображения к ближайшей степени двойки
rows, cols = img.shape
new_rows = next_power_of_2(rows)
new_cols = next_power_of_2(cols)

# Дополнение изображения до нужного размера
img_padded = np.zeros((new_rows, new_cols), dtype=np.complex64)
img_padded[:rows, :cols] = img

# Применение БПФ (FFT) вручную
def fft_iterative(x):
    N = len(x)
    # Проводим битовую перестановку для упорядочивания индексов
    bit_reversed = np.array([int(bin(i)[2:].zfill(int(np.log2(N)))[::-1], 2) for i in range(N)], dtype=int)
    x = x[bit_reversed]

    # Итеративная обработка
    step = 1
    while step < N:
        # Множители для текущего шага
        W = np.exp(-2j * np.pi * np.arange(step) / (2 * step))
        for k in range(0, N, 2 * step):
            for j in range(step):
                t = W[j] * x[k + j + step]
                x[k + j + step] = x[k + j] - t
                x[k + j] += t
        step *= 2
    return x

# Применяем БПФ к изображению
def fft2d(img):
    # Применяем БПФ по строкам
    for i in range(img.shape[0]):
        img[i, :] = fft_iterative(img[i, :])
    # Применяем БПФ по столбцам
    for i in range(img.shape[1]):
        img[:, i] = fft_iterative(img[:, i])
    return img

# Применяем БПФ (FFT)
image_freq = fft2d(img_padded)

# Логарифмическое преобразование спектра для улучшения видимости
magnitude_spectrum = np.log(np.abs(image_freq) + 1)

# Показать спектр
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (Manual FFT)')
plt.show()

# Восстановление изображения с помощью обратного БПФ
def ifft_iterative(x):
    N = len(x)
    # Проводим битовую перестановку для упорядочивания индексов
    bit_reversed = np.array([int(bin(i)[2:].zfill(int(np.log2(N)))[::-1], 2) for i in range(N)], dtype=int)
    x = x[bit_reversed]

    # Итеративная обработка с положительным знаком в экспоненте
    step = 1
    while step < N:
        # Множители для текущего шага (с положительным знаком в экспоненте)
        W = np.exp(2j * np.pi * np.arange(step) / (2 * step))  # Знак изменен на положительный
        for k in range(0, N, 2 * step):
            for j in range(step):
                t = W[j] * x[k + j + step]
                x[k + j + step] = x[k + j] - t
                x[k + j] += t
        step *= 2
    return x

# Применяем обратный БПФ (IFFT)
def ifft2d(img):
    # Применяем обратный БПФ по строкам
    for i in range(img.shape[0]):
        img[i, :] = ifft_iterative(img[i, :])
    # Применяем обратный БПФ по столбцам
    for i in range(img.shape[1]):
        img[:, i] = ifft_iterative(img[:, i])
    return img

# Восстанавливаем изображение с нормализацией
restored_img = ifft2d(image_freq)
restored_img = np.real(restored_img) / (new_rows * new_cols)  # Нормализация по размеру изображения

# Показать восстановленное изображение
plt.imshow(restored_img[:rows, :cols], cmap='gray')
plt.title('Restored Image (Manual IFFT)')
plt.show()
