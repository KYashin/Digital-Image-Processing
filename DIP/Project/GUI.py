from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow
import sys

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("Цифровая обработка изображений")

    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText("My First Label!")
        self.label.move(50, 50)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Click me")
        self.b1.clicked.connect(self.clicked)

    def clicked(self):
        self.label.setText("You pressed the button")

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.showMaximized()
    sys.exit(app.exec())

window()