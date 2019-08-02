import time
import numpy as np
import cv2.cv2 as cv2
import qdarkstyle
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication,QSplashScreen
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QListWidget, QAction, qApp, QMenu,QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QSizePolicy
from Database import Database
from add_window.SetCamera import SetCamera
from add_window.AddPicworkForm import AddPicworkForm
from add_window.Setsetting import SetSetting
from processor.MainProcessor import MainProcessor
from yolo3.yolo import YOLO
from yolo3.yolo_online import YOLO_ONLINE
from add_window.Statistics import AddstatisForm
from add_window.help import HelpForm
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
'18':'空织'
}
class MainWindow(QMainWindow):
    def __init__(self,model_flag='local'):
        self.model_flag = model_flag
        super(MainWindow, self).__init__()
        loadUi("./UI/MainForm.ui", self)
        self.cam_01 = False
        self.cam_02 = False
        self.cam_03 = False
        self.run_flag = False
        self.vs1 = None
        self.feed1 = None
        self.vs2 = None
        self.feed2 = None
        self.vs3 = None
        self.feed3 = None
        self.cnt = -1
        res = QMessageBox.question(self,'提示','是否使用在线识别模式？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if res == QMessageBox.Yes:
            self.model_flag = 'online'
        else:
            self.model_flag = 'local'
        if (self.model_flag=='local'):
            self.yolo = YOLO()
        elif (self.model_flag == 'online'):
            self.yolo = YOLO_ONLINE()
        self.st = time.time()
        self.coco = False
        self.first = True

        self.CameraA.setScaledContents(True)
        self.CameraA.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.CameraB.setScaledContents(True)
        self.CameraB.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.CameraC.setScaledContents(True)
        self.CameraC.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.cam_clear_gaurd = False

        self.processor = MainProcessor(['cam_01','cam_02','cam_03'])


        self.database = Database.getInstance()

        #----------数据库初始化-------------------------------------#
        # self.database.insertIntoCamera('cam_01')
        # self.database.insertIntoCamera('cam_02')
        # self.database.insertIntoCamera('cam_03')
        # self.database.insertIntoSetting()
        # for i in range(19)[1:]:
        #     flow_type=class_name[str(i)]
        #     self.database.insertIntoFlawStatistic(flaw_type=flow_type,flaw_cont=0)
        # self.database.deleteAllFlaw()
        #----------------------------------------------------------#

        #----------测试数据-----------------------------------------#
        # for i in range(19)[1:]:
        #     self.database.insertIntoFlaw(flaw_id=i-1,flaw_type=class_name[str(i)])
        # for i in range(19)[1:]:
        #     self.database.updateFlawStatistic(flaw_type=class_name[str(i)],flaw_cont=1)
        #----------------------------------------------------------#

        self.statistic_cnt = {}
        for i in range(19)[1:]:
            self.statistic_cnt[class_name[str(i)]]=self.database.getcntFromFlawStatistic(flaw_type=class_name[str(i)])
        self.database.updataCaminfo('cam_01')
        self.database.updataCaminfo('cam_02')
        self.database.updataCaminfo('cam_03')
        self.database.updataOpenflag('cam_01',0)
        self.database.updataOpenflag('cam_02',0)
        self.database.updataOpenflag('cam_03',0)
        self.ID = self.database.getFlawCount() #获取最大的标号

        self.updateCamInfo()

        self.PB_CamAsettings.clicked.connect(self.setCameraA)
        self.PB_CamBsettings.clicked.connect(self.setCameraB)
        self.PB_CamCsettings.clicked.connect(self.setCameraC)
        self.PB_picture.clicked.connect(self.picworkForm)
        self.PB_start.clicked.connect(self.start)
        self.PB_end.clicked.connect(self.stop)
        self.User.triggered.connect(self.helpform)
        self.PB_statistics.clicked.connect(self.statisForm)
        self.PB_settings.clicked.connect(self.setting)

        self.tablemodel = QStandardItemModel(10, 8)
        self.tablemodel.setHorizontalHeaderLabels(['编号','瑕疵种类','相机','X坐标','Y坐标','瑕疵宽度','瑕疵高度','时间戳'])
        self.tV_info.setModel(self.tablemodel)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(50)
        
    def setCameraA(self):
        if self.run_flag == False:
            addWin = SetCamera(parent=self,camid='cam_01')
            addWin.show()
        else:
            res = QMessageBox.question(self,'警告','请先停止实时识别')

    def setCameraB(self):
        if self.run_flag == False:
            addWin = SetCamera(parent=self,camid='cam_02')
            addWin.show()
        else:
            res = QMessageBox.question(self,'警告','请先停止实时识别')
    def setCameraC(self):
        if self.run_flag == False:
            addWin = SetCamera(parent=self,camid='cam_03')
            addWin.show()
        else:
            res = QMessageBox.question(self,'警告','请先停止实时识别')
    def picworkForm(self):
        #self.yolo.close_session()
        if (self.model_flag=='local'):
            addWin = AddPicworkForm(parent=self,yolo=self.yolo,model_flag=self.model_flag)
        else:
            addWin = AddPicworkForm(parent=self,yolo=self.yolo,model_flag=self.model_flag)
        addWin.show()
    def helpform(self):
        addWin = HelpForm(parent=self)
        addWin.show()
    def statisForm(self):
        for i in range(19)[1:]:
            self.database.updateFlawStatistic(flaw_type=class_name[str(i)],flaw_cont=self.statistic_cnt[class_name[str(i)]])
        addWin = AddstatisForm(parent=self)
        addWin.show()
    def setting(self):
        addWin = SetSetting(parent=self)
        addWin.show()
    def update_image(self):
        try:
            if self.run_flag == True:
                frames = {}
                frames[0]=None
                frames[1]=None
                frames[2]=None
                if self.cam_01 == False and self.cam_02 == False and self.cam_03 == False:
                    return
                if self.cam_01 == True:
                    ret1, frame1 = self.vs1.read()
                    if ret1 == True:
                        frames[0]=frame1
                if self.cam_02 == True:
                    ret2, frame2 = self.vs2.read()
                    if ret2 == True:
                        frames[1]=frame2
                if self.cam_03 == True:
                    ret3, frame3 = self.vs3.read()
                    if ret3 == True:
                        frames[2]=frame3
                packet = self.processor.getProcessedImage(yolo = self.yolo,frames = frames)
                boxes1 = packet['boxes1']
                boxes2 = packet['boxes2']
                boxes3 = packet['boxes3']
                if self.cam_01 and isinstance(packet['frame01'],np.ndarray):
                    qimg = self.toQImage(packet['frame01'])
                    self.CameraA.setPixmap(QPixmap.fromImage(qimg))
                if self.cam_02 and isinstance(packet['frame02'],np.ndarray):
                    qimg = self.toQImage(packet['frame02'])
                    self.CameraB.setPixmap(QPixmap.fromImage(qimg))
                if self.cam_03 and isinstance(packet['frame03'],np.ndarray):
                    qimg = self.toQImage(packet['frame03'])
                    self.CameraC.setPixmap(QPixmap.fromImage(qimg))
                if self.cam_01:
                    for i in range(len(boxes1['label'])):
                        ID = self.ID
                        self.ID += 1
                        label = class_name[boxes1['label'][i]]
                        if self.coco:
                            label = boxes1['label'][i]
                        X = boxes1['X'][i]
                        Y = boxes1['Y'][i]
                        W = boxes1['W'][i]
                        H = boxes1['H'][i]
                        camer_id = 'cam_01'
                        timeStamp = int(time.time())
                        timeArray = time.localtime(timeStamp)
                        flaw_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
                        rate = Database.getInstance().getSetting()
                        wid_value = rate[0]
                        hight_value = rate[1]
                        rate_value = rate[2]
                        ft_time = time.time()
                        Y = (ft_time-self.st)*rate_value + Y
                        if(W>=wid_value and H>=hight_value):
                            self.identify_work(ID=ID, label=label, camera_id=camer_id, X=X, Y=Y, W=W, H=H, flaw_time=flaw_time)
                            self.database.insertIntoFlaw(flaw_id=int(ID),
                                                        flaw_type=label,
                                                        camera_id=camer_id,
                                                        coordinate_x=X,
                                                        coordinate_y=Y,
                                                        width=W,
                                                        highth=H,
                                                        flaw_time=flaw_time)
                            self.statistic_cnt[label]+=1
                            print('ID: '+str(ID))
                if self.cam_02:
                    for i in range(len(boxes2['label'])):
                        ID = self.ID
                        self.ID += 1
                        label = class_name[boxes2['label'][i]]
                        if self.coco:
                            label = boxes2['label'][i]
                        X = boxes2['X'][i]
                        Y = boxes2['Y'][i]
                        W = boxes2['W'][i]
                        H = boxes2['H'][i]
                        camer_id = 'cam_02'
                        timeStamp = int(time.time())
                        timeArray = time.localtime(timeStamp)
                        flaw_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
                        rate = Database.getInstance().getSetting()
                        wid_value = rate[0]
                        hight_value = rate[1]
                        rate_value = rate[2]
                        ft_time = time.time()
                        Y = (ft_time-self.st)*rate_value + Y
                        if(W>=wid_value and H>=hight_value):
                            self.identify_work(ID=ID, label=label, camera_id=camer_id, X=X, Y=Y, W=W, H=H, flaw_time=flaw_time)
                            self.database.insertIntoFlaw(flaw_id=int(ID),
                                                        flaw_type=label,
                                                        camera_id=camer_id,
                                                        coordinate_x=X,
                                                        coordinate_y=Y,
                                                        width=W,
                                                        highth=H,
                                                        flaw_time=flaw_time)
                            self.statistic_cnt[label]+=1
                            print('ID: '+str(ID))
                if self.cam_03:
                    for i in range(len(boxes3['label'])):
                        ID = self.ID
                        self.ID += 1
                        label = class_name[boxes3['label'][i]]
                        if self.coco:
                            label = boxes3[label][i]
                        X = boxes3['X'][i]
                        Y = boxes3['Y'][i]
                        W = boxes3['W'][i]
                        H = boxes3['H'][i]
                        camer_id = 'cam_03'
                        timeStamp = int(time.time())
                        timeArray = time.localtime(timeStamp)
                        flaw_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
                        rate = Database.getInstance().getSetting()
                        wid_value = rate[0]
                        hight_value = rate[1]
                        rate_value = rate[2]
                        ft_time = time.time()
                        Y = (ft_time-self.st)*rate_value + Y
                        if(W>=wid_value and H>=hight_value):
                            self.identify_work(ID=ID, label=label, camera_id=camer_id, X=X, Y=Y, W=W, H=H, flaw_time=flaw_time)
                            self.database.insertIntoFlaw(flaw_id=int(ID),
                                                        flaw_type=label,
                                                        camera_id=camer_id,
                                                        coordinate_x=X,
                                                        coordinate_y=Y,
                                                        width=W,
                                                        highth=H,
                                                        flaw_time=flaw_time)
                            self.statistic_cnt[label]+=1
                            print('ID: '+str(ID))
        
        except:
            res = QMessageBox.question(self,'警告','链接网络摄像头失败,请核对摄像头参数')
            self.run_flag=False
            

    def identify_work(self, ID = 0, label = "0", camera_id = "cam_01", X = 1.0, Y = 1.0, W = 1.0, H = 1.0, flaw_time = 0):
        ID = str(ID)
        label = str(label)
        camera_id = str(camera_id)
        X = str(X)
        Y = str(Y)
        W = str(W)
        H = str(H)
        flaw_time = str(flaw_time)
        self.cnt += 1
        self.sm=self.tablemodel
        self.sm.setItem(self.cnt, 0, QStandardItem(ID))         
        self.sm.setItem(self.cnt, 1, QStandardItem(label))
        self.sm.setItem(self.cnt, 2, QStandardItem(camera_id))
        self.sm.setItem(self.cnt, 3, QStandardItem(X))
        self.sm.setItem(self.cnt, 4, QStandardItem(Y))
        self.sm.setItem(self.cnt, 5, QStandardItem(W))
        self.sm.setItem(self.cnt, 6, QStandardItem(H))
        self.sm.setItem(self.cnt, 7, QStandardItem(flaw_time))
        self.tV_info.setModel(self.sm)
        #QTableView
        self.tV_info.setColumnWidth(0,100)
        self.tV_info.setColumnWidth(1,200)        
            
          

    def updateCamInfo(self):
        self.feed1 = self.database.getCamurl('cam_01')[0]
        self.feed2 = self.database.getCamurl('cam_02')[0]
        self.feed3 = self.database.getCamurl('cam_03')[0]
        self.cam_01 = self.database.getOpenflag('cam_01')[0]
        self.cam_02 = self.database.getOpenflag('cam_02')[0]
        self.cam_03 = self.database.getOpenflag('cam_03')[0]
        if self.feed1 == "":
            self.feed1 = 0
        if self.feed2 == "":
            self.feed2 = 0
        if self.feed3 == "":
            self.feed3 = 0
        #self.processor = MainProcessor(self.cam_selector.currentText())
        if self.first==False:
            self.vs1 = cv2.VideoCapture(self.feed1)
            self.vs2 = cv2.VideoCapture(self.feed2)
            self.vs3 = cv2.VideoCapture(self.feed3)
        else:
            self.first = False

    #def updateLog(self):
    def updateCamFlagInfo(self):
        self.cam_01 = self.database.getOpenflag('cam_01')[0]
        self.cam_02 = self.database.getOpenflag('cam_02')[0]
        self.cam_03 = self.database.getOpenflag('cam_03')[0]



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
    def start(self):
        if self.run_flag == True :
            return
        self.updateCamFlagInfo()
        #self.database.updataOpenflag('cam_01',1)
        #self.database.updataOpenflag('cam_02',1)
        #self.database.updataOpenflag('cam_03',1)
        if(self.cam_01 or self.cam_02 or self.cam_03):
            ash = QSplashScreen(QtGui.QPixmap("UI/NUI.png"))
            ash.setFont(QtGui.QFont('Microsoft YaHei UI',20))
            ash.show()
            ash.showMessage("摄像头链接中,请稍候",QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.green)
            self.updateCamInfo()
            self.run_flag = True
            if (self.cam_01 == True and self.vs1.isOpened() == False) or (self.cam_02 == True and self.vs2.isOpened() == False) or (self.cam_03 == True and self.vs3.isOpened() == False):
                self.run_flag = False
                ash.close()
                res = QMessageBox.question(self,'警告','摄像头链接失败,请核对摄像头参数')
            else:
                ash.close()
               
        else:
            res = QMessageBox.question(self,'警告','请先设置摄像头参数')

    
    


    @QtCore.pyqtSlot()
    def stop(self):
        if self.run_flag == False:
            return
        self.run_flag = False
        if (self.vs1 != None):
            self.vs1.release()
            self.CameraA.setText("已停止")
        if (self.vs2 != None):
            self.vs2.release()
            self.CameraB.setText("已停止")
        if (self.vs3 != None):
            self.vs3.release()
            self.CameraC.setText("已停止")
        self.vs1 = None
        self.vs2 = None
        self.vs3 = None
        res = QMessageBox.question(self,'提示','是否保留摄像头设置？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if res == QMessageBox.Yes:
            return
        else:
            self.database.updataOpenflag('cam_01',0)
            self.database.updataOpenflag('cam_02',0)
            self.database.updataOpenflag('cam_03',0)
            #self.updateCamInfo()
            self.cam_01 = False
            self.cam_02 = False
            self.cam_03 = False

    def closeEvent(self,QCloseEvent):
        res = QMessageBox.question(self,'提示','是否退出系统？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if res == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


        