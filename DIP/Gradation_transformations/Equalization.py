import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalization(path_to_image):
    # Загружаем изображение
    image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)

    lst = np.zeros(256)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            intensity = image[y, x]
            lst[intensity] += 1

    for i in range(len(lst)):
        lst[i] += lst[i - 1]

    cdf_m = np.ma.masked_equal(lst, 0)
    cdf_m = (cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min()) * 255
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    equalized_image = cdf[image]

    orig_hist = calculate_histogram(image)
    equalized_hist = calculate_histogram(equalized_image)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Увеличиваем размер графиков

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')  # Убираем оси

    axes[0, 1].imshow(equalized_image, cmap='gray')
    axes[0, 1].set_title('Changed Image', fontsize=14)
    axes[0, 1].axis('off')  # Убираем оси

    # Гистограмма для оригинального изображения
    axes[1, 0].bar(range(256), orig_hist, color='yellow', width=1.0, edgecolor="black")
    axes[1, 0].set_title('Histogram of Original Image', fontsize=14)
    axes[1, 0].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Гистограмма для измененного изображения
    axes[1, 1].bar(range(256), equalized_hist, color='red', width=1.0, edgecolor="black")
    axes[1, 1].set_title('Histogram of Changed Image', fontsize=14)
    axes[1, 1].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Выравниваем графики
    plt.tight_layout()
    plt.show()

def calculate_histogram(img):
    # histogram = [0] * 256                # действие вложенного цикла аналогично тому, что делает функция np.bincount
    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):
    #         intensity = img[y, x]
    #         histogram[intensity] += 1

    # Подсчет количества пикселей каждого значения интенсивности
    histogram = np.bincount(img.ravel(), minlength=256)
    # Нормализация гистограммы
    normalized_hist = histogram / img.size
    return normalized_hist

equalization(r"D:\pythonProject\DIP\Project\Images\Lenna_test_image.png")