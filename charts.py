import os
import sys
import cv2
import torch
from PySide6.QtCore import QFileInfo
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, \
    QFileDialog, QStackedWidget, QComboBox
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtCore import Qt, QTimer, QUrl


class Chart_win(QMainWindow):
    def __init__(self):
        super().__init__()

        # 窗口大小和标题
        self.outputDir=os.getcwd()+"/output/"
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("The Count of Object")
        self.icon=QIcon("img/gongda.jpg")
        self.change_icon()
        # 创建主窗口部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.stackWidget = QStackedWidget(self)
        self.vLayout2 = QVBoxLayout()
        self.detect=QPushButton('检测目标数量统计')
        self.detect.setMaximumWidth(30)
        self.detect.setMaximumWidth(180)
        self.browser = QWebEngineView(self)
        self.browser.setStyleSheet('''QWebEngineView{background-color:rgb(104,205,210)}''')
        self.vLayout2.addWidget(self.detect)
        self.vLayout2.addWidget(self.browser)
        central_widget.setLayout(self.vLayout2)
        self.detect.clicked.connect(self.view)

        # 绑定按钮点击事件
    def view(self):
        if os.path.exists(self.outputDir + 'chart/count.html'):
            self.browser.load(QUrl(QFileInfo(self.outputDir + 'chart/count.html').absoluteFilePath()))

    def change_icon(self):
        """用来修改图像的图标"""
        self.setWindowIcon(self.icon)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = Chart_win()
#     window.show()
#     sys.exit(app.exec())






