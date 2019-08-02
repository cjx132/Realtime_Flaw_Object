from PyQt5.QtWidgets import QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets
#from Database import Database
from add_window.AddMainWindow import AddMainWindow
import imghdr
import os.path
import cv2.cv2 as cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from yolo3.yolo import YOLO
class_name = {
'1':'停车痕紧',
'2':'停车痕松',
'3':'断经',
'4':'错花',
'5':'并纬',
'6':'缩纬',
'7':'缺纬',
'8':'糙纬',
'9':'折返',
'10':'断纬',
'11':'油污',
'12':'起机',
'13':'尽机',
'14':'经条',
'15':'擦白',
'16':'擦伤',
'17':'浆斑',
'18':'空织',
'19':'缺纬'
}
def file_extension(path):
    return os.path.splitext(path)[1]

class AddPicworkForm(AddMainWindow):
    def __init__(self, parent=None,yolo=None,model_flag='local'):
        super().__init__(parent, "UI/PicworkForm.ui")
        self.YOLO = yolo
        self.model_flag=model_flag
        self.shape = (640,480)
        self.path = ''
        self.cnt = -1
        self.type = 'error'

        self.tablemodel = QStandardItemModel(10,5)
        self.tablemodel.setHorizontalHeaderLabels(['瑕疵种类','X坐标','Y坐标','宽度W','高度H'])
        #self.tablemodels = QStandardItemModel(10,5)
        #self.tablemodels.setHorizontalHeaderLabels(['瑕疵种类','X坐标','Y坐标','宽度W','高度H'])
        self.TV_info.setModel(self.tablemodel)
        
        # self.add.clicked.connect(self.addToDatabase)
        self.cancel.clicked.connect(self.close)
        self.file_browse.clicked.connect(lambda: self.getFile(self.file))
        self.PB_work.clicked.connect(self.work)
    def close(self):
        #self.YOLO.close_session()
        self.destroy(True)

    def addToDatabase(self):
        pass
    def getFile(self, lineEdit):
        pic_map = QFileDialog.getOpenFileName()[0]
        self.path = pic_map
        lineEdit.setText(pic_map)
        # print(pic_map)
        if pic_map!="":
            if imghdr.what(pic_map)!=None:
                pixmap = QPixmap (pic_map)
                self.LB_Pic.setPixmap(pixmap)
                self.LB_Pic.setScaledContents(True)
                self.LB_Pic.show()
                self.type = 'pic'
            elif file_extension(pic_map) in ['.AVI','.avi','.MP4','.mp4','.mov','.MOV','.rmvb','.RMVB','.wmv','.WMV']:
                vid = cv2.VideoCapture(pic_map)
                if not vid.isOpened():
                    raise IOError("Couldn't open webcam or video")
                return_value, frame = vid.read()
                frame = cv2.resize(frame,(480,640))
                qimg = self.toQImage(frame)
                self.LB_Pic.setPixmap(QPixmap.fromImage(qimg))
                self.type = 'mov'
                vid.release()
            else:
                self.type = 'error'




    
    def identify_work(self, label = "0", X = 1.0, Y = 1.0, W = 1.0, H = 1.0):
        label = str(label)
        X = str(X)
        Y = str(Y)
        W = str(W)
        H = str(H)
        self.cnt += 1
        self.sm=self.tablemodel         
        self.sm.setItem(self.cnt, 0, QStandardItem(label))
        self.sm.setItem(self.cnt, 1, QStandardItem(X))
        self.sm.setItem(self.cnt, 2, QStandardItem(Y))
        self.sm.setItem(self.cnt, 3, QStandardItem(W))
        self.sm.setItem(self.cnt, 4, QStandardItem(H))
        self.TV_info.setModel(self.sm)
        #QTableView
        self.TV_info.setColumnWidth(0,100)
        self.TV_info.setColumnWidth(1,200)

    
    def toQImage(self, raw_img):
        from numpy import copy
        img = copy(raw_img)
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img.tobytes(), img.shape[1], img.shape[0], img.strides[0], qformat)
        outImg = outImg.rgbSwapped()
        return outImg

    @QtCore.pyqtSlot()
    def work(self):
        if self.type == 'pic':
            frame = cv2.imread(self.path)
            self.cnt = -1
            self.tablemodel = QStandardItemModel(10,5)
            self.tablemodel.setHorizontalHeaderLabels(['瑕疵种类','X坐标','Y坐标','宽度W','高度H'])
            self.TV_info.setModel(self.tablemodel)
            try:
                frame = cv2.resize(frame,(480,640))
                pack = self.object_detecation_pic(frame)
                frame = pack['frame']
                box = pack['box']
                for i in range(len(box['label'])):
                    self.identify_work(class_name[box['label'][i]], box['X'][i], box['Y'][i], box['W'][i], box['H'][i])
                qimg = self.toQImage(frame)
                self.LB_Pic.setPixmap(QPixmap.fromImage(qimg))
            except:
                print("image error")
        elif self.type == 'mov':
            vid = cv2.VideoCapture(self.path)
            self.cnt = -1
            self.tablemodel = QStandardItemModel(10,5)
            self.tablemodel.setHorizontalHeaderLabels(['瑕疵种类','X坐标','Y坐标','宽度W','高度H'])
            self.TV_info.setModel(self.tablemodel)
            while True:
                return_value, frame = vid.read()
                if return_value:
                    try:
                        frame = cv2.resize(frame,(480,640))
                        pack = self.object_detecation_pic(frame)
                        frame = pack['frame']
                        box = pack['box']
                        for i in range(len(box['label'])):
                            self.identify_work(class_name[box['label'][i]], box['X'][i], box['Y'][i], box['W'][i], box['H'][i])
                        qimg = self.toQImage(frame)
                        self.LB_Pic.setPixmap(QPixmap.fromImage(qimg))
                    except:
                        print("image error")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            vid.release()
        else:
            self.LB_Pic.setText("请选择图片或视频")





    
    def object_detecation_pic(self,frame):
        ans={}
        if (isinstance(frame,np.ndarray)):        
            #image = cv2.resize(frame,self.shape)
            image = Image.fromarray(frame)
            image,ans = self.YOLO.detect_image(image)
            #image,ans = yolo.detect_image(image)
            frame = np.asarray(image)
        else:
            frame = None
        pack = {'frame': frame,'box':ans}
        return pack

    def closeEvent(self,QCloseEvent):
        res = QMessageBox.question(self,'提示','是否退出？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if res == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()  

    # def addToDatabase(self):
    #     id = str(self.id.text())
    #     group = str(self.group.text())
    #     location = str(self.location.text())
    #     x = str(self.x_coord.text())
    #     y = str(self.y_coord.text())
    #     file = str(self.file.text())
    #     Database.getInstance().insertIntoCamera(id, location, x, y, group, file)
    #     self.destroy()
