import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загружаем изображение
image = cv2.imread(r"C:\Users\user\PycharmProjects\Digital-Image-Processing\DIP\Images_DIP\4.webp", cv2.IMREAD_GRAYSCALE)
image_1 = cv2.imread(r"C:\Users\user\PycharmProjects\Digital-Image-Processing\DIP\Images_DIP\5.webp", cv2.IMREAD_GRAYSCALE)

# Добавляем шум
noise = np.random.normal(0, 10, image_1.shape).astype(np.float32)
noisy_image = cv2.add(image_1.astype(np.float32), noise)

# Применяем Guided Filter
guided_image = image.copy()  # То же изображение используется в качестве направляющего
filtered_image = cv2.ximgproc.guidedFilter(
    image, noisy_image, radius=100, eps=1e-3
)

# Визуализация
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_1, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Filtered Image")
plt.imshow(filtered_image, cmap="gray")
plt.axis("off")

plt.show()