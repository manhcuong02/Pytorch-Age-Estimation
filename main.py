import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer, QRect
import os
from predict import AgeEstimator
from PIL import Image, ImageQt
import numpy as np
import cv2 as cv


class AgeEstimationGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.window_height = 800
        self.window_width = 1600

        # Setup window
        self.setWindowTitle('Age Estimation GUI')
        self.setGeometry(100, 100, self.window_width, self.window_height)
        self.setFixedSize(self.window_width, self.window_height)

        # Initialize variables
        self.weights = None
        self.model = AgeEstimator(weights=self.weights, device='cuda')
        self.timer = QTimer()
        self.cap = None
        self.webcam_active = False

        # QLabel for images and webcam feed
        self.in_image = QLabel(self)
        self.in_image.setGeometry(50, 150, 640, 480)
        self.in_image.setStyleSheet("background-color: black;")
        self.in_image.setAlignment(Qt.AlignCenter)
        self.in_image.hide()  # Hide QLabel initially

        self.img_path = None
        self.out_image = QLabel(self)
        self.out_image.hide()

        # Combobox
        self.select_mode = self.add_ComboBox(50, 50, ["Please select a mode", "Webcam", "Image"], event=self.on_mode_selected)
        self.select_weight = self.add_ComboBox(500, 50, ["Please select a weights"] + self.get_weights_path(), event=self.on_weights_selected)

        # Buttons
        self.webcam_button = self.add_button("Connect to Webcam", 300, 700, 190, 50, self.Connect_to_Webcam, mode='hide')
        self.select_image_button = self.add_button("Select Image", 320, 700, 150, 50, self.selectImage, mode='hide')
        self.exit_button = self.add_button("Exit", 1280, 700, 150, 50, exit)
        self.save_button = self.add_button("Save Result", 960, 700, 150, 50, self.save_result, mode='hide')
        self.predict_button = self.add_button("Predict Image", 640, 700, 150, 50, self.predict, mode='hide')

    def on_weights_selected(self, index):
        weights_path = self.select_weight.itemText(index)
        if weights_path == "Please select a weights":
            self.weights = None
        else:
            self.weights = weights_path

        self.model = AgeEstimator(weights=self.weights, device='cuda')

    def get_weights_path(self):
        weights_filename = os.listdir("weights")
        weights_path = [os.path.join(os.getcwd(), "weights", i) for i in weights_filename]
        return weights_path

    def Connect_to_Webcam(self):
        if not self.webcam_active:
            # Start the webcam
            self.clear_display()  # Clear any previously uploaded images
            self.cap = cv.VideoCapture(0)  # 0 for default webcam
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                return

            # Connect QTimer to update_frame
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # Update every 30ms
            self.in_image.show()  # Show the QLabel for webcam feed
            self.webcam_active = True
            self.webcam_button.setText("Stop Webcam")
        else:
            # Stop the webcam
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.webcam_active = False
            self.webcam_button.setText("Connect to Webcam")
            self.clear_display()  # Clear the display when stopping the webcam

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            return

        # Convert BGR to RGB for processing
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        try:
            # Process with AgeEstimator
            processed_frame = self.model.predict_frame(rgb_frame)

            # Convert processed frame to QPixmap
            height, width, channel = processed_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Update QLabel to show the processed frame
            self.in_image.setPixmap(pixmap)
        except Exception as e:
            print(f"Error during frame processing: {e}")

    def on_mode_selected(self, index):
        self.mode = self.select_mode.itemText(index)
        if self.mode == "Webcam":
            self.clear_display()  # Clear images when switching to webcam
            self.select_image_button.hide()
            self.webcam_button.show()
            self.save_button.show()
            self.predict_button.hide()
            self.in_image.show()  # Show QLabel for webcam
        elif self.mode == "Please select a mode":
            self.select_image_button.hide()
            self.webcam_button.hide()
            self.save_button.hide()
            self.predict_button.hide()
            self.clear_display()
        elif self.mode == "Image":
            if self.webcam_active:
                self.timer.stop()
                if self.cap:
                    self.cap.release()
                self.webcam_active = False
                self.webcam_button.setText("Connect to Webcam")
            self.clear_display()  # Clear webcam feed when switching to image mode
            self.select_image_button.show()
            self.webcam_button.hide()
            self.save_button.show()
            self.predict_button.show()

    def clear_display(self):
        """Clears all displays."""
        self.in_image.clear()
        self.in_image.hide()  # Hide QLabel for webcam
        self.out_image.clear()
        self.out_image.hide()  # Hide QLabel for output

    def predict(self):
        if self.img_path:
            self.predicted_image = self.model.predict(self.img_path)
            qt_img = Image.fromarray(self.predicted_image, mode='RGB')
            qt_img = ImageQt.ImageQt(qt_img)
            pixmap = QPixmap.fromImage(qt_img)
            width, height = self.rescale_image(self.predicted_image.shape[1], self.predicted_image.shape[0])
            self.out_image.setGeometry(QRect(1200 - width // 2, 150, width, height))
            self.out_image.setPixmap(pixmap.scaled(width, height))
            self.out_image.show()  # Show QLabel for output

    def save_result(self):
        if self.predicted_image is not None:
            img = cv.cvtColor(self.predicted_image, cv.COLOR_RGB2BGR)
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Result", "", "Images (*.png *.jpg)")
            if file_path:
                cv.imwrite(file_path, img)

    def add_ComboBox(self, x, y, items: list, event=None, font_size=10):
        combo = QComboBox(self)
        for i in items:
            combo.addItem(i)
        combo.move(x, y)
        font = QFont()
        font.setPointSize(font_size)
        combo.setFont(font)
        if event:
            combo.currentIndexChanged.connect(event)
        return combo

    def add_button(self, title, x, y, w, h, event=None, font_size=10, mode='show'):
        assert mode in ['hide', 'show']
        button = QPushButton(self)
        button.setText(title)
        button.move(x, y)
        button.setFixedSize(w, h)
        font = QFont()
        font.setPointSize(font_size)
        button.setFont(font)
        if mode == 'show':
            button.show()
        else:
            button.hide()
        if event:
            button.clicked.connect(event)
        return button

    def rescale_image(self, width, height):
        return int(width * 500 / height), 500

    def selectImage(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)')
        if file_name:
            self.img_path = file_name
            pixmap = QPixmap(file_name)
            img = cv.imread(file_name)
            width, height = self.rescale_image(img.shape[1], img.shape[0])
            self.in_image.setGeometry(QRect(400 - width // 2, 150, width, height))
            self.in_image.setPixmap(pixmap.scaled(width, height))
            self.in_image.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AgeEstimationGUI()
    window.show()
    sys.exit(app.exec_())
