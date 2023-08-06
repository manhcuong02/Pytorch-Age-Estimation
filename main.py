import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QPoint, QRect
import os
from predict import Model
from PIL import Image, ImageQt
import numpy as np
import cv2 as cv

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.window_heigt = 800
        self.window_width = 1600
        
        self.img_width = 700
        self.img_height = 400
                
        # Thiết lập tiêu đề và kích thước của cửa sổ
        self.setWindowTitle('Image Viewer')
        self.setGeometry(100, 100, self.window_width, self.window_heigt)
        self.setFixedSize(self.window_width, self.window_heigt)
        
        # Combobox
        self.mode = None
        self.select_mode = self.add_ComboBox(50,50, ["Please select a mode", "Webcam", "Image"], event = self.on_mode_selected)
        
        self.select_weight = self.add_ComboBox(250,50, ["Please select a weights"] + self.get_weights_path(), event = self.on_weights_selected)    
        self.weights = None
        
        # model
        self.model = Model(weights = self.weights, device = 'cuda')
        
        # 
        self.in_image = QLabel(self)
        self.in_image.setGeometry(QRect(50, 200, 700, 400))
        self.img_path = None
        # 
        self.out_image = QLabel(self)
        self.out_image.setGeometry(QRect(850, 200, 700, 400))
        self.predicted_image = None
        
        self.webcam_button = self.add_button("Connect to Webcam", 300, 700, 190, 50, self.Connect_to_Webcam, mode = 'hide')
        self.select_image_button = self.add_button("Select Image", 320, 700, 150, 50, self.selectImage, mode = 'hide')
        self.exit_button = self.add_button("Exit", 1280, 700, 150, 50, exit)
        self.save_button = self.add_button("Save Result", 960, 700, 150, 50, self.save_result, mode = 'hide')
        self.predict_button = self.add_button("Predict Image", 640, 700, 150, 50, self.predict, mode = 'hide')
        
    def on_weights_selected(self, index):
        weights_path = self.select_weight.itemText(index)
        if weights_path == "Please select a weights":
            self.weights = None
        else:
            self.weights = weights_path
        
        self.model = Model(weights = self.weights, device = 'cuda')
        
    def get_weights_path(self):
        weights_filename = os.listdir("weights")
        weights_path = [os.path.join(os.getcwd() , "weights", i)for i in weights_filename]
        return weights_path
    
    def Connect_to_Webcam(self):
        pass
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.gray, 3))
        painter.drawLine(800,200,800,600)

    def on_mode_selected(self, index):
        self.mode = self.select_mode.itemText(index)
        if self.mode == "Webcam":
            self.select_image_button.hide()
            self.webcam_button.show()
            self.save_button.show()
            self.predict_button.show()
            
        elif self.mode == "Please select a mode":
            self.select_image_button.hide()
            self.webcam_button.hide()
            self.save_button.hide()
            self.predict_button.hide()
        
        elif self.mode == "Image":
            self.select_image_button.show()
            self.webcam_button.hide()
            self.save_button.show()
            self.predict_button.show()

    def add_ComboBox(self, x, y, items : list, event = None, font_size = 10):
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
        
    def predict(self):
        if self.img_path:
            self.predicted_image = self.model.predict(self.img_path)
            qt_img = Image.fromarray(self.predicted_image, mode = 'RGB')
            qt_img = ImageQt.ImageQt(qt_img)
            pixmap = QPixmap.fromImage(qt_img)
            width, height = self.rescale_image(self.predicted_image.shape[1], self.predicted_image.shape[0])
            self.out_image.setPixmap(pixmap.scaled(width, height))    
            
    def save_result(self):
        # Chụp nội dung của QWidget
        if self.predicted_image is not None:
            img = cv.cvtColor(self.predicted_image, cv.COLOR_RGB2BGR)
            # Lưu ảnh thành file
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Result", "", "Images (*.png *.jpg)")
            if file_path:
                cv.imwrite(file_path, img)
        
    def add_button(self, title, x, y, w, h, event = None, font_size = 10, mode = 'show'):
        assert mode in ['hide', 'show']
        button = QPushButton(self)
        button.setText(title)
        button.move(x,y)
        button.setFixedSize(w,h)
        
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
        if width < height:
            return int(width * self.img_height / height), self.img_height

        else:
            return self.img_width, int(height * self.img_width/width)

    def selectImage(self):
        # Hiển thị hộp thoại chọn tệp ảnh và lấy tên tệp ảnh được chọn
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)')
        
        if file_name:
            self.img_path = file_name
            # Tải ảnh từ tệp và hiển thị nó trên QLabel
            pixmap = QPixmap(file_name)
            
            width, height = self.rescale_image(self.in_image.width(), self.in_image.height())
            
            self.in_image.setPixmap(pixmap.scaled(width, height))        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec_())
