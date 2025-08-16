import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QFileDialog,
                             QFrame, QSizePolicy)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QColor, QPalette
from PyQt5.QtCore import Qt, QCoreApplication, QSize
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR


class ShipLicenseRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(
            det_model_dir='model/det',
            rec_model_dir='model/rec',
            use_angle_cls=False,
            use_gpu=True
        )

    def initUI(self):
        # 设置主窗口属性
        self.setWindowTitle('船牌检测与识别系统')
        self.setWindowIcon(QIcon('icon/ship_icon.png'))
        self.setFixedSize(1400, 900)

        # 设置主窗口背景渐变
        self.setAutoFillBackground(True)
        palette = self.palette()
        gradient = "qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #1a2a6c, stop:1 #21a1f1)"
        palette.setBrush(QPalette.Window, QColor(240, 245, 250))  # 更柔和的背景色
        self.setPalette(palette)

        # 创建主布局
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # 左侧面板 - 原始图像显示
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 2px solid #d0d0d0;
            }
        """)
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)

        self.image_label = QLabel("请上传包含船牌的图片", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #666;
                background-color: white;
                border-radius: 10px;
            }
        """)
        left_layout.addWidget(self.image_label)

        # 右侧面板 - 检测结果
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 15px;
                border: 2px solid #d0d0d0;
            }
        """)
        right_panel.setFixedWidth(450)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(20)

        # 标题
        title_label = QLabel("船牌识别结果", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                border-bottom: 2px solid #3498db;
            }
        """)
        right_layout.addWidget(title_label)

        # 检测结果区域
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        result_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)

        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(15, 15, 15, 15)
        result_layout.setSpacing(15)

        # 船牌检测标题
        self.label_text = QLabel("检测到的船牌区域", self)
        self.label_text.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #3498db;
            }
        """)
        result_layout.addWidget(self.label_text)

        # 船牌图像显示
        self.plate_image_label = QLabel("等待检测...", self)
        self.plate_image_label.setAlignment(Qt.AlignCenter)
        self.plate_image_label.setStyleSheet("""
            QLabel {
                min-height: 150px;
                background-color: #eef2f7;
                border-radius: 8px;
                border: 1px dashed #a0a0a0;
                font-size: 14px;
                color: #7f8c8d;
            }
        """)
        result_layout.addWidget(self.plate_image_label)

        # 识别结果标题
        self.label_text1 = QLabel("识别结果", self)
        self.label_text1.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #3498db;
            }
        """)
        result_layout.addWidget(self.label_text1)

        # 船牌号码显示
        self.plate_number_label = QLabel("--", self)
        self.plate_number_label.setAlignment(Qt.AlignCenter)
        self.plate_number_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                background-color: #e8f4fc;
                border-radius: 8px;
                padding: 15px;
                border: 1px solid #d6eaf8;
            }
        """)
        result_layout.addWidget(self.plate_number_label)

        right_layout.addWidget(result_frame)

        # 按钮区域
        button_frame = QFrame()
        button_layout = QVBoxLayout(button_frame)
        button_layout.setSpacing(15)
        button_layout.setContentsMargins(0, 0, 0, 0)

        # 上传按钮
        upload_button = QPushButton('上传图片', self)
        upload_button.setIcon(QIcon('icon/upload-icon.png'))
        upload_button.setIconSize(QSize(24, 24))
        upload_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a5276;
            }
        """)
        upload_button.clicked.connect(self.uploadImage)
        button_layout.addWidget(upload_button)

        # 退出按钮
        exit_button = QPushButton("退出系统", self)
        exit_button.setIcon(QIcon('icon/exit-icon.png'))
        exit_button.setIconSize(QSize(24, 24))
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #922b21;
            }
        """)
        exit_button.clicked.connect(QCoreApplication.instance().quit)
        button_layout.addWidget(exit_button)

        right_layout.addWidget(button_frame)
        right_layout.addStretch()

        # 将左右面板添加到主布局
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

        # 创建主窗口中心的主Widget
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 设置窗口阴影效果
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #f5f7fa, stop:1 #c3cfe2);
            }
        """)

        self.show()

    def uploadImage(self):
        # 打开文件对话框，选择图片文件
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("图片文件 (*.png *.jpg *.bmp)")
        image_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', 'Image Files (*.png *.jpg *.bmp)')

        if image_path:
            # 加载图片并显示在左侧Label中
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

            # 更新状态
            self.plate_image_label.setText("检测中...")
            self.plate_number_label.setText("识别中...")
            QApplication.processEvents()  # 强制更新UI

            # 进行船牌检测与识别
            plate_image, plate_number = self.detectAndRecognizeShipLicense(image_path)

            # 在右侧Label中显示切割出来的船牌图片和识别结果
            if plate_image is not None:
                self.plate_image_label.setPixmap(QPixmap.fromImage(plate_image).scaled(
                    self.plate_image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
            else:
                self.plate_image_label.setText("未检测到船牌")
                self.plate_image_label.setStyleSheet("""
                    QLabel {
                        min-height: 150px;
                        background-color: #fde8e8;
                        border-radius: 8px;
                        border: 1px dashed #e74c3c;
                        font-size: 14px;
                        color: #c0392b;
                    }
                """)

            self.plate_number_label.setText(plate_number if plate_number else "未能识别船牌号码")

    def detectAndRecognizeShipLicense(self, image_path):
        # 读取图像
        img = cv2.imread(image_path)

        # 使用PaddleOCR进行检测和识别
        result = self.ocr.ocr(img, cls=False)

        # 如果没有检测到任何文本区域，返回None
        if not result or not result[0]:
            return None, ""

        # 获取所有检测框和识别结果
        boxes = [line[0] for line in result[0]]
        texts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        # 找出最可能是船牌的文本区域（这里简单取第一个或得分最高的）
        best_idx = scores.index(max(scores)) #0
        if len(scores) > 1:
            best_idx = scores.index(max(scores))

        best_box = boxes[best_idx]
        best_text = texts[best_idx]

        # 裁剪船牌区域
        x_coords = [int(p[0]) for p in best_box]
        y_coords = [int(p[1]) for p in best_box]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        # 确保坐标在图像范围内
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 裁剪船牌区域
        plate_img = img[y1:y2, x1:x2]

        # 将OpenCV图像转换为Qt图像
        if plate_img.size > 0:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            h, w, ch = plate_img.shape
            bytes_per_line = ch * w
            from PyQt5.QtGui import QImage
            q_img = QImage(plate_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            q_img = None

        return q_img, best_text


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 设置应用程序字体
    font = QFont()
    font.setFamily("Microsoft YaHei")  # 使用微软雅黑字体
    font.setPointSize(10)
    app.setFont(font)

    mainWindow = ShipLicenseRecognitionApp()
    mainWindow.show()
    sys.exit(app.exec_())