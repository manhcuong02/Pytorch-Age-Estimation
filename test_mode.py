# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QVBoxLayout, QWidget

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("Mode Selector")
#         self.setGeometry(100, 100, 400, 300)

#         # Tạo QWidget và QVBoxLayout cho MainWindow
#         central_widget = QWidget(self)
#         self.setCentralWidget(central_widget)
#         layout = QVBoxLayout(central_widget)

#         # Tạo QComboBox để chọn chế độ
#         self.mode_selector = QComboBox(self)
#         self.mode_selector.addItem("Mode 1")
#         self.mode_selector.addItem("Mode 2")
#         layout.addWidget(self.mode_selector)

#         # Khi chọn chế độ, gọi hàm on_mode_selected
#         self.mode_selector.currentIndexChanged.connect(self.on_mode_selected)

#         # Tạo QLabel để hiển thị giao diện theo chế độ
#         self.interface_label = QLabel(self)
#         layout.addWidget(self.interface_label)

#     def on_mode_selected(self, index):
#         # Khi chọn chế độ, cập nhật giao diện dựa vào chế độ được chọn
#         selected_mode = self.mode_selector.itemText(index)
#         if selected_mode == "Mode 1":
#             self.interface_label.setText("Chế độ 1 được chọn. Hiển thị giao diện cho chế độ 1.")
#         elif selected_mode == "Mode 2":
#             self.interface_label.setText("Chế độ 2 được chọn. Hiển thị giao diện cho chế độ 2.")

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())

