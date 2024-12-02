import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def sharpening(path_to_image, laplas_filter, alpha=1.0):
    img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)

    kernel = laplas_filter

    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2

    padded_image = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    filtered_image = np.zeros_like(img, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            region = padded_image[y:y + kh, x:x + kw]
            filtered_image[y, x] = np.sum(region * kernel)

    normalized_image = normalize_image(filtered_image)

    sharpened_image = np.clip(img + alpha * normalized_image, 0, 255)

    show_histograms_and_images(img, normalized_image, sharpened_image)


def normalize_image(image, min_out=-128, max_out=128):
    # Находим минимальное и максимальное значение в изображении
    min_in = np.min(image)
    max_in = np.max(image)

    # Применяем линейную нормализацию по формуле
    normalized_image = (image - min_in) / (max_in - min_in) # Масштабируем в диапазон [0, 1]
    normalized_image = normalized_image * (max_out - min_out) + min_out  # Масштабируем в диапазон [min_out, max_out]

    return normalized_image

def show_histograms_and_images(orig_img, filtered_img, sharpened_image):
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))  # Увеличиваем размер графиков

    axes[0, 0].imshow(orig_img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')  # Убираем оси

    axes[0, 1].imshow(filtered_img, cmap='gray')
    axes[0, 1].set_title('Filtered Image', fontsize=14)
    axes[0, 1].axis('off')  # Убираем оси

    axes[0, 2].imshow(sharpened_image, cmap='gray')
    axes[0, 2].set_title('Filtered Image', fontsize=14)
    axes[0, 2].axis('off')  # Убираем оси

    # Гистограмма для оригинального изображения
    axes[1, 0].hist(orig_img.ravel(), bins=256, range=(0, 255), color='red', alpha=0.75)
    axes[1, 0].set_title('Histogram of Original Image', fontsize=14)
    axes[1, 0].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Гистограмма для фильтрованного изображения
    axes[1, 1].hist(filtered_img.ravel(), bins=256, range=(0, 255), color='red', alpha=0.75)
    axes[1, 1].set_title('Histogram of Changed Image', fontsize=14)
    axes[1, 1].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Гистограмма для измененного изображения
    axes[1, 2].hist(sharpened_image.ravel(), bins=256, range=(0, 255), color='red', alpha=0.75)
    axes[1, 2].set_title('Histogram of Changed Image', fontsize=14)
    axes[1, 2].set_xlabel('Pixel Intensity', fontsize=12)
    axes[1, 2].set_ylabel('Frequency', fontsize=12)
    axes[1, 2].grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку

    # Выравниваем графики
    plt.tight_layout()
    plt.show()

filter_laplas_1 = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])

filter_laplas_2 = np.array([[0, -1, 0],
                            [-1,  4, -1],
                            [0, -1, 0]])

sharpening(r"C:\Users\user\PycharmProjects\PythonProject\DIP\Project\Images\Lenna_test_image.png", filter_laplas_1) # 1231231



