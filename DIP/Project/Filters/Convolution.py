import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r"D:\pythonProject\DIP\Project\Images\Lenna_test_image.png", cv.IMREAD_GRAYSCALE)

# kernel = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) / 11

kernel = np.array([[-1, 0, -1],
                   [0,  4, 0],
                   [-1, 0, -1]])

h, w = img.shape
kh, kw = kernel.shape
pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2

padded_image = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

filtered_image = np.zeros_like(img, dtype=np.float32)

for y in range(h):
    for x in range(w):
        region = padded_image[y:y + kh, x:x + kw]
        filtered_image[y, x] = np.sum(region * kernel)

filtered_image = cv.normalize(filtered_image, None, alpha=-128, beta=128, norm_type=cv.NORM_MINMAX)

alpha = 1.0
final_image = np.clip(img + alpha * filtered_image, 0, 255)

fig, axes = plt.subplots(1, 3, figsize=(10, 6))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image', fontsize=14)
axes[0].axis('off')  # Убираем оси

axes[1].imshow(filtered_image, cmap='gray')
axes[1].set_title('Filtered Image', fontsize=14)
axes[1].axis('off')  # Убираем оси

axes[2].imshow(final_image, cmap='gray')
axes[2].set_title('Sharpened Image', fontsize=14)
axes[2].axis('off')  # Убираем оси

plt.tight_layout()
plt.show()