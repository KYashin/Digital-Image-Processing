from PyQt6 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.showMaximized()

        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.button1 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.button1.setGeometry(QtCore.QRect(280, 380, 151, 51))
        self.button1.setObjectName("button1")
        self.label1 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(300, 110, 141, 41))
        self.label1.setObjectName("label1")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        # Подключаем обработчик кнопки
        self.button1.clicked.connect(self.on_button_click)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button1.setText(_translate("MainWindow", "Press me"))
        self.label1.setText(_translate("MainWindow", "Hello, my name is Kirill!"))

    def on_button_click(self):
        # Вызываем функцию brightness при нажатии кнопки
        image_path = r"D:\pythonProject\DIP\Images_DIP\Lenna_test_image.png"
        coefficient = 100
        self.brightness(image_path, coefficient)

    def brightness(self, path_to_image, coefficient):
        # Считываем изображение в градациях серого
        img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Image not found!")
            return

        img_1 = img.copy()

        print(f"Original shape: {img_1.shape}")

        img_1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        for y in range(img_1.shape[0]):
            for x in range(img_1.shape[1]):
                img_1[y, x] = img[y, x] + coefficient
                if coefficient < 0:
                    if img_1[y, x] < 0:
                        img_1[y, x] = 0
                elif coefficient > 0:
                    if img_1[y, x] > 255:
                        img_1[y, x] = 255

        img_1 = img_1.astype(np.uint8)

        orig_hist = self.calculate_histogram(img)
        changed_hist = self.calculate_histogram(img_1)

        self.show_histograms_and_images(img, img_1, orig_hist, changed_hist)

    def calculate_histogram(self, img):
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

    def show_histograms_and_images(self, orig_img, changed_img, orig_hist, changed_hist):

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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
