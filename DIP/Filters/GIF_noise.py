import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загружаем изображение
image = cv2.imread(r"D:\pythonProject\DIP\Images_DIP\Lenna_test_image.png", cv2.IMREAD_GRAYSCALE)
image_1 = cv2.imread(r"D:\pythonProject\DIP\Images_DIP\4.png", cv2.IMREAD_GRAYSCALE)

# Добавляем шум
noise = np.random.normal(0, 10, image.shape).astype(np.float32)
noisy_image = cv2.add(image.astype(np.float32), noise)

# Применяем Guided Filter
guided_image = image.copy()  # То же изображение используется в качестве направляющего
filtered_image = cv2.ximgproc.guidedFilter(
    image, image_1, radius=1, eps=1e-3
)

# Визуализация
plt.figure(figsize=(12, 6))
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
plt.imshow(filtered_image, cmap="gray")
plt.axis("off")

plt.show()