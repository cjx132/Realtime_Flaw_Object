from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
class AddMainWindow(QMainWindow):
    def __init__(self, parent=None, ui=None):
        super(AddMainWindow, self).__init__(parent)
        loadUi(ui, self)

        self.tablemodel = QStandardItemModel(10,6)
        self.tablemodel.setHorizontalHeaderLabels(['ID','瑕疵种类','X坐标','Y坐标','时间'])
        self.TV_info.setModel(self.tablemodel)
        
        # self.add.clicked.connect(self.addToDatabase)
        self.cancel.clicked.connect(self.close)
        self.file_browse.clicked.connect(lambda: self.getFile(self.file))
        self.PB_work.clicked.connect(self.identify_work)
    def close(self):
        self.destroy(True)

    def addToDatabase(self):
        pass
    def getFile(self, lineEdit):
        pic_map = QFileDialog.getOpenFileName()[0]
        lineEdit.setText(pic_map)
        # print(pic_map)
        pixmap = QPixmap (pic_map)
        self.LB_Pic.setPixmap(pixmap)
        self.LB_Pic.setScaledContents(True)
        self.LB_Pic.show()
    
    def identify_work(self):
        # def __init__(self, parent=None):
        #     super(MainWindow, self).__init__(parent)
        #     loadUi('qtdesigner.ui', self)
        #     self.pushButton.clicked.connect(self.say)
        #     self.showData()
        # def showData(self):
        #     # 准备数据模型
        self.sm=self.tablemodel           
        #设置数据头栏名称
        # self.sm.setHorizontalHeaderItem(0, QtGui.QStandardItem("Name"))
        # self.sm.setHorizontalHeaderItem(1, QtGui.QStandardItem("NO."))
        
        #设置数据条目
        self.sm.setItem(0, 0, QStandardItem("刘董"))
        self.sm.setItem(0, 1, QStandardItem("123"))
        
        self.sm.setItem(1, 0, QStandardItem("杜阳阳"))
        self.sm.setItem(1, 1, QStandardItem("456"))
        
        self.sm.setItem(2, 0, QStandardItem("常家鑫"))
        self.sm.setItem(2, 1, QStandardItem("789"))
        
        #设置条目颜色和字体
        # self.sm.item(0, 0).setForeground(QBrush(QColor(255, 0, 0)))		
        # self.sm.item(0, 0).setFont(QFont("Times", 10, QFont.Black))
        
        # self.sm.item(3, 1).setBackground(QBrush(QColor(255, 255, 0)))
        
        #按照编号排序
        # self.sm.sort(1,Qt.DescendingOrder)
        
        #将数据模型绑定到QTableView
        self.TV_info.setModel(self.sm)
        
        #QTableView
        self.TV_info.setColumnWidth(0,100)
        self.TV_info.setColumnWidth(1,200)
        