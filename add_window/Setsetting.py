from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtCore, QtWidgets
from Database import Database
from add_window.AddMainWindow import AddMainWindow


class SetSetting(AddMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent, "UI/Setting.ui")
        #self.file_browse.clicked.connect(lambda: self.getFile(self.file))
        self.setok.clicked.connect(self.updataSetToDatabase)
        self.clearall.clicked.connect(self.clearSetToDatabase)
        # caminfo = Database.getInstance().getCaminfo(ID=camid)[0]
        self.hight.setText('')
        self.wid.setText('')
        self.rate_video.setText('')
        self.model.setText('yolo')
        model_name = str(self.model.text())
        rate = Database.getInstance().getSetting(model=model_name)
        wid_value = rate[0]
        hight_value = rate[1]
        rate_value = rate[2] 
        self.hight.setText(str(hight_value))
        self.wid.setText(str(wid_value))
        self.rate_video.setText(str(rate_value))

    @QtCore.pyqtSlot()
    def clearSetToDatabase(self):
        model_name = str(self.model.text())
        Database.getInstance().updateSetting(model_name)
        rate = Database.getInstance().getSetting(model=model_name)
        wid_value = rate[0]
        hight_value = rate[1]
        rate_value = rate[2] 
        self.hight.setText(str(hight_value))
        self.wid.setText(str(wid_value))
        self.rate_video.setText(str(rate_value))
        #self.model.setText('')

    @QtCore.pyqtSlot()
    def updataSetToDatabase(self):
        model_name = str(self.model.text())
        hight_value = str(self.hight.text())
        wid_value = str(self.wid.text())
        rate_value = str(self.rate_video.text())
        Database.getInstance().updateSetting(model_name,wid_value,hight_value,rate_value)
    
    def closeEvent(self,QCloseEvent):
        res = QMessageBox.question(self,'提示','是否保存设置？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if res == QMessageBox.Yes:
            model_name = str(self.model.text())
            hight_value = str(self.hight.text())
            wid_value = str(self.wid.text())
            rate_value = str(self.rate_video.text())
            Database.getInstance().updateSetting(model_name,wid_value,hight_value,rate_value)
            QCloseEvent.accept()
        else:
            QCloseEvent.accept()
        

