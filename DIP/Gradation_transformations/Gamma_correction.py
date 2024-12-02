import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def gamma_correction(path_to_image, gamma):
    img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)

    img_1 = np.zeros((img.shape[0], img.shape[1]))
    for y in range(img_1.shape[0]):
        for x in range(img_1.shape[1]):
            img_1[y, x] = 255 * (img[y, x] / 255) ** gamma

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

gamma_correction(r"D:\pythonProject\DIP\Project\Images\i.webp", 0.5)