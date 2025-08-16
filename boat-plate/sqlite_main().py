import sys
import sqlite3  # 导入sqlite3模块
import datetime  # 用于时间戳
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QFileDialog,
                             QFrame, QSizePolicy, QMessageBox)  # 添加QMessageBox
from PyQt5.QtGui import QPixmap, QFont, QIcon, QColor, QPalette
from PyQt5.QtCore import Qt, QCoreApplication, QSize
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR


class ShipLicenseRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_name = None  # 用于存储当前图片的文件名
        self.current_recognized_text = None  # 用于存储当前识别的文本

        self.initUI()
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(
            det_model_dir='model/det',
            rec_model_dir='model/rec',
            use_angle_cls=False,
            use_gpu=True  # 请确保您的环境支持GPU并且PaddleOCR GPU版本已正确安装
        )
        self.init_db()  # 初始化数据库

    def init_db(self):
        """初始化数据库连接和表"""
        self.db_conn = sqlite3.connect('ship_recognition_data.db')
        self.db_cursor = self.db_conn.cursor()
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                recognized_text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.db_conn.commit()

    def initUI(self):
        # 设置主窗口属性
        self.setWindowTitle('船牌检测与识别系统')
        self.setWindowIcon(QIcon('icon/ship_icon.png'))  # 请确保图标文件存在
        self.setFixedSize(1400, 900)

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setBrush(QPalette.Window, QColor(240, 245, 250))
        self.setPalette(palette)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

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
                background-color: white; /* 改为white与面板一致 */
                border-radius: 10px;
            }
        """)
        left_layout.addWidget(self.image_label)

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

        self.label_text = QLabel("检测到的船牌区域", self)
        self.label_text.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #3498db;
            }
        """)
        result_layout.addWidget(self.label_text)

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

        self.label_text1 = QLabel("识别结果", self)
        self.label_text1.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #3498db;
            }
        """)
        result_layout.addWidget(self.label_text1)

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

        button_frame = QFrame()
        button_layout = QVBoxLayout(button_frame)
        button_layout.setSpacing(15)
        button_layout.setContentsMargins(0, 0, 0, 0)

        upload_button = QPushButton('上传图片', self)
        upload_button.setIcon(QIcon('icon/upload-icon.png'))  # 请确保图标文件存在
        upload_button.setIconSize(QSize(24, 24))
        upload_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db; color: white; font-size: 18px; font-weight: bold;
                padding: 12px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1a5276; }
        """)
        upload_button.clicked.connect(self.uploadImage)
        button_layout.addWidget(upload_button)

        # 新增“保存到数据库”按钮
        self.save_db_button = QPushButton('保存到数据库', self)
        # self.save_db_button.setIcon(QIcon('icon/save_icon.png')) # 可选：添加保存图标
        self.save_db_button.setIconSize(QSize(24, 24))
        self.save_db_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71; color: white; font-size: 18px; font-weight: bold;
                padding: 12px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #27ae60; }
            QPushButton:pressed { background-color: #1f8b4c; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d;}
        """)
        self.save_db_button.clicked.connect(self.saveToDatabase)
        self.save_db_button.setEnabled(False)  # 初始时禁用
        button_layout.addWidget(self.save_db_button)

        exit_button = QPushButton("退出系统", self)
        exit_button.setIcon(QIcon('icon/exit-icon.png'))  # 请确保图标文件存在
        exit_button.setIconSize(QSize(24, 24))
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; color: white; font-size: 18px; font-weight: bold;
                padding: 12px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:pressed { background-color: #922b21; }
        """)
        exit_button.clicked.connect(self.close)  # 改为self.close以触发closeEvent
        button_layout.addWidget(exit_button)

        right_layout.addWidget(button_frame)
        right_layout.addStretch()

        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #f5f7fa, stop:1 #c3cfe2);
            }
        """)
        self.show()

    def uploadImage(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("图片文件 (*.png *.jpg *.bmp)")
        image_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', 'Image Files (*.png *.jpg *.bmp)')

        if image_path:
            self.current_image_name = os.path.basename(image_path)  # 获取文件名
            self.current_recognized_text = None  # 重置当前识别文本
            self.save_db_button.setEnabled(False)  # 上传新图片后，先禁用保存按钮

            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),  # 使用实际大小
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

            self.plate_image_label.setText("检测中...")
            self.plate_image_label.setStyleSheet("""
                QLabel {
                    min-height: 150px; background-color: #eef2f7; border-radius: 8px;
                    border: 1px dashed #a0a0a0; font-size: 14px; color: #7f8c8d;
                }
            """)
            self.plate_number_label.setText("识别中...")
            QApplication.processEvents()

            plate_image_qimg, plate_number = self.detectAndRecognizeShipLicense(image_path)

            if plate_image_qimg is not None:
                self.plate_image_label.setPixmap(QPixmap.fromImage(plate_image_qimg).scaled(
                    self.plate_image_label.width(), self.plate_image_label.height(),  # 使用实际大小
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
            else:
                self.plate_image_label.setText("未检测到船牌")
                self.plate_image_label.setStyleSheet("""
                    QLabel {
                        min-height: 150px; background-color: #fde8e8; border-radius: 8px;
                        border: 1px dashed #e74c3c; font-size: 14px; color: #c0392b;
                    }
                """)

            if plate_number:
                self.plate_number_label.setText(plate_number)
                self.current_recognized_text = plate_number  # 存储识别结果
                self.save_db_button.setEnabled(True)  # 识别成功，激活保存按钮
            else:
                self.plate_number_label.setText("未能识别船牌号码")
                self.current_recognized_text = None
                self.save_db_button.setEnabled(False)

    def detectAndRecognizeShipLicense(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, ""

        result = self.ocr.ocr(img, cls=False)

        if not result or not result[0]:
            return None, ""

        boxes = [line[0] for line in result[0]]
        texts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        if not scores:  # 以防万一scores为空
            return None, ""

        # 找出得分最高的文本区域
        best_idx = scores.index(max(scores))
        best_box = boxes[best_idx]
        best_text = texts[best_idx]

        x_coords = [int(p[0]) for p in best_box]
        y_coords = [int(p[1]) for p in best_box]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x1 >= x2 or y1 >= y2:  # 确保裁剪区域有效
            return None, best_text  # 即使无法裁剪，也可能识别出了文本

        plate_img_cv = img[y1:y2, x1:x2]

        q_img = None
        if plate_img_cv.size > 0:
            try:
                plate_img_rgb = cv2.cvtColor(plate_img_cv, cv2.COLOR_BGR2RGB)
                h, w, ch = plate_img_rgb.shape
                bytes_per_line = ch * w
                from PyQt5.QtGui import QImage
                q_img = QImage(plate_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            except cv2.error as e:
                print(f"OpenCV error during color conversion: {e}")
                return None, best_text  # 返回已识别文本，即使图像处理失败

        return q_img, best_text

    def saveToDatabase(self):
        """将当前识别结果保存到数据库"""
        if self.current_image_name and self.current_recognized_text:
            try:
                self.db_cursor.execute(
                    "INSERT INTO recognitions (image_filename, recognized_text) VALUES (?, ?)",
                    (self.current_image_name, self.current_recognized_text)
                )
                self.db_conn.commit()
                QMessageBox.information(self, "成功", "数据已成功保存到数据库！")
                self.save_db_button.setEnabled(False)  # 保存后禁用，避免重复保存同一条记录
            except sqlite3.Error as e:
                QMessageBox.warning(self, "数据库错误", f"保存数据失败: {e}")
        else:
            QMessageBox.warning(self, "无数据", "没有可保存的图片或识别结果。")

    def closeEvent(self, event):
        """关闭窗口前关闭数据库连接"""
        if self.db_conn:
            self.db_conn.close()
        super().closeEvent(event)


if __name__ == '__main__':
    #QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # 支持高DPI
    app = QApplication(sys.argv)

    font = QFont()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(10)
    app.setFont(font)

    mainWindow = ShipLicenseRecognitionApp()
    mainWindow.show()
    sys.exit(app.exec_())