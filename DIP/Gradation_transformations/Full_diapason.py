import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def full(path_to_image):

    img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)

    # Приведение к полному диапазону [0, 255] с использованием NumPy
    img_min = np.min(img)
    img_max = np.max(img)
    img_1 = 255 * ((img - img_min) / (img_max - img_min))
    img_1 = img_1.astype(np.uint8)

    orig_hist = calculate_histogram(img)
    changed_hist = calculate_histogram(img_1)

    show_histograms_and_images(img, img_1, orig_hist, changed_hist)

def calculate_histogram(img):

    # Подсчет количества пикселей каждого значения интенсивности
    histogram = np.bincount(img.ravel(), minlength=256)
    # Нормализация гистограммы
    normalized_hist = histogram / img.size
    return normalized_hist


def show_histograms_and_images(orig_img, changed_img, orig_hist, changed_hist):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Увеличиваем размер графиков

    axes[0, 0].imshow(orig_img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')  # Убираем оси

    axes[0, 1].imshow(changed_img, cmap='gray')
    axes[0, 1].set_title('Changed Image', fontsize=14)
    axes[0, 1].axis('off')  # Убираем оси

    # Гистограмма для оригинального изображения
    axes[1, 0].bar(range(256), orig_hist, color='yellow', width=1.0, edgecolor="black")
    axes[1, 0].set_title('Histogram of Original Image', fontsize=14)
    axes[1, 0].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Гистограмма для измененного изображения
    axes[1, 1].bar(range(256), changed_hist, color='red', width=1.0, edgecolor="black")
    axes[1, 1].set_title('Histogram of Changed Image', fontsize=14)
    axes[1, 1].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Выравниваем графики
    plt.tight_layout()
    plt.show()


full(r'D:\pythonProject\DIP\Images_DIP\cf50445f-7887-44a2-ab11-8407f5be1ccc.jpg')

