from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import testModel

dict = ['洗手盆', '马桶', '插座', '开关', '小便器', '阀门']

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.label1 = QLabel(self)
        self.labelAns = QLabel(self)
        okButton = QPushButton('打开图片')
        cancelButton = QPushButton('分析预测')
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)
        hbox.addStretch(1)
        global vbox
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addStretch(1)
        self.setLayout(vbox)
        okButton.clicked.connect(self.openfile)
        okButton.clicked.connect(self.labelshow)
        cancelButton.clicked.connect(self.ansShow)
        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle('构件识别')
        self.show()

    def openfile(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        global path_openfile_name
        path_openfile_name = openfile_name[0]
        self.labelAns.setText('')

    def labelshow(self):
        self.labelPic = QtWidgets.QLabel(self)
        self.labelPic.setPixmap(QPixmap(""))
        if len(path_openfile_name) > 0:
            self.labelPic.move(100, 45)
            self.labelPic.setMinimumSize(100, 100)
            self.labelPic.setPixmap(QPixmap(path_openfile_name))
            self.labelPic.setScaledContents(True)
        self.labelPic.show();

    def ansShow(self):
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        font.setBold(True)
        self.label1.setText('')
        self.label1.setText('经过缜密分析，这个构件应该是——')
        self.label1.move(20, 150)
        self.label1.setMinimumWidth(500)
        self.label1.setFont(font)
        #self.label1.setStyleSheet("")
        self.label1.show()
        num = testModel.classify(path_openfile_name)
        self.labelAns.setText(dict[num])
        self.labelAns.setFont(font)
        self.labelAns.setStyleSheet("color:blue")
        self.labelAns.move(120, 180)
        self.labelAns.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())