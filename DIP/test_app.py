from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton

class TestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test App")
        self.resize(400, 300)
        self.button = QPushButton("Open Image", self)
        self.button.clicked.connect(self.open_image)
        self.button.setGeometry(100, 100, 200, 50)

    def open_image(self):
        print(1)

        print(type(self))
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть изображение",
            "",
            "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)"
        )
        print(2)
        if file_path:
            print(f"Выбран файл: {file_path}")
        else:
            print("Файл не выбран")

if __name__ == "__main__":
    app = QApplication([])
    window = TestApp()
    window.show()
    app.exec()