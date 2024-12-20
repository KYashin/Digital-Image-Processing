import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def logarithm(path_to_image, base):
    img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)

    image_normalized = img / 255.0

    # Преобразуем изображение в логарифмическое
    c = 1.0 / np.log(1 + np.max(image_normalized))  # Коэффициент нормализации
    img_1 = c * (np.log(1 + image_normalized) / np.log(base))

    img_1 = np.uint8(255 * img_1)

    show_histograms_and_images(img, img_1)

def show_histograms_and_images(orig_img, changed_img):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Увеличиваем размер графиков

    axes[0, 0].imshow(orig_img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')  # Убираем оси

    axes[0, 1].imshow(changed_img, cmap='gray')
    axes[0, 1].set_title('Changed Image', fontsize=14)
    axes[0, 1].axis('off')  # Убираем оси

    # Гистограмма для оригинального изображения
    # axes[1, 0].bar(range(256), [intensity / max(orig_hist) for intensity in orig_hist], color='yellow', width=1.0, edgecolor="black")
    axes[1, 0].hist(orig_img.ravel(), bins=256, range=(0, 255), color='red', alpha=0.75)
    axes[1, 0].set_title('Histogram of Original Image', fontsize=14)
    axes[1, 0].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Гистограмма для измененного изображения
    # axes[1, 1].bar(range(256), [intensity / max(changed_hist) for intensity in changed_hist], color='red', width=1.0, edgecolor="black")
    axes[1, 1].hist(changed_img.ravel(), bins=256, range=(0, 255), color='red', alpha=0.75)
    axes[1, 1].set_title('Histogram of Changed Image', fontsize=14)
    axes[1, 1].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Выравниваем графики
    plt.tight_layout()
    plt.show()

logarithm(r"C:\Users\user\PycharmProjects\Digital-Image-Processing\DIP\Images_DIP\mountains-8567074_640.webp", 0.5)