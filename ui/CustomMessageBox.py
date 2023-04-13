from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QPixmap, QIcon


# 单按钮对话框，出现到指定时长后自动消失
class MessageBox(QMessageBox):
    def __init__(self, *args, title='提示', count=1, time=1000, auto=False, **kwargs):
        super(MessageBox, self).__init__(*args, **kwargs)
        self._count = count
        self._time = time
        self.icon=QIcon('img/gongda.jpg')
        self._auto = auto  # 是否自动关闭
        assert count > 0  # 必须大于0
        assert time >= 500  # 必须>=500毫秒
        self.setStyleSheet('''
                            QDialog{background:rgb(75, 75, 75);
                                    color:white;}
                            QLabel{color:white;
                                    background: rgb(75, 75, 75);
                                    font-size: 15px;
                                    font-weight: light;
                                    color:white;}''')

        self.setWindowTitle(title)
        self.setWindowIcon(self.icon)
        self.setIconPixmap(QPixmap("img/face.png"))

        self.setStandardButtons(self.Close)  # 关闭按钮
        self.closeBtn = self.button(self.Close)  # 获取关闭按钮
        self.closeBtn.setText('关闭')
        self.closeBtn.setVisible(False)
        self._timer = QTimer(self, timeout=self.doCountDown)
        self._timer.start(self._time)

    def doCountDown(self):
        # self.closeBtn.setText('关闭(%s)' % self._count)
        self._count -= 1
        if self._count <= 0:
            # self.closeBtn.setText('关闭')
            # self.closeBtn.setEnabled(True)
            self._timer.stop()
            if self._auto:  # 自动关闭
                self.accept()
                self.close()


if __name__ == '__main__':
    MessageBox(QWidget=None, text='123', auto=True).exec_()
