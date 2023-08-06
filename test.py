import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import cv2

class ImageViewer(QMainWindow):
    def __init__(self, image_array):
        super().__init__()

        # Tạo QLabel để hiển thị ảnh
        self.image_label = QLabel()

        # Chuyển numpy array thành QImage
        height, width, channel = image_array.shape
        qimage = QImage(image_array.data, width, height, width * channel, QImage.Format_RGB888)

        # Hiển thị ảnh lên QLabel
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

        # Tạo layout và thêm QLabel vào layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # Tạo nút để lưu ảnh
        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        layout.addWidget(save_button)

        # Tạo QWidget để chứa layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def save_image(self):
        # Chụp nội dung của QWidget
        pixmap = self.centralWidget().grab()

        # Chuyển QPixmap thành numpy array
        image_array = np.array(pixmap.toImage())

        # Lưu ảnh thành file
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg)")
        if file_path:
            cv2.imwrite(file_path, cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR))

if __name__ == '__main__':
    # Tạo numpy array ảnh (vd: ảnh màu 100x100)
    image_array = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    app = QApplication(sys.argv)
    window = ImageViewer(image_array)
    window.show()
    sys.exit(app.exec_())
