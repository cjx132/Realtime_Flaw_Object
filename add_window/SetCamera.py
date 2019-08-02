from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from Database import Database
from add_window.AddMainWindow import AddMainWindow


class SetCamera(AddMainWindow):
    def __init__(self, parent=None,camid='cam_01'):
        self.clear = True
        super().__init__(parent, "UI/SetCamera.ui")
        #self.file_browse.clicked.connect(lambda: self.getFile(self.file))
        self.ID.setText(camid)
        self.set.clicked.connect(self.updataCaminfoToDatabase)
        self.clear.clicked.connect(self.clearCaminfoToDatabase)
        caminfo = Database.getInstance().getCaminfo(ID=camid)[0]
        self.X.setText(str(caminfo[6]))
        self.Y.setText(str(caminfo[7]))
        self.rate.setText(str(caminfo[5]))
        self.IP.setText(str(caminfo[0]))
        self.Port.setText(str(caminfo[1]))
        self.Username.setText(str(caminfo[2]))
        self.Password.setText(str(caminfo[3]))

    @QtCore.pyqtSlot()
    def clearCaminfoToDatabase(self):
        self.clear = True
        camid=str(self.ID.text())
        Database.getInstance().updataCaminfo(camid=camid)
        Database.getInstance().updataOpenflag(camid,0)
        caminfo = Database.getInstance().getCaminfo(ID=camid)[0]
        self.X.setText(str(caminfo[6]))
        self.Y.setText(str(caminfo[7]))
        self.rate.setText(str(caminfo[5]))
        self.IP.setText(str(caminfo[0]))
        self.Port.setText(str(caminfo[1]))
        self.Username.setText(str(caminfo[2]))
        self.Password.setText(str(caminfo[3]))

    
    @QtCore.pyqtSlot()
    def updataCaminfoToDatabase(self):
        self.clear = False
        camid=str(self.ID.text())
        coordinate_x=str(self.X.text())
        coordinate_y=str(self.Y.text())
        rate=str(self.rate.text())
        camip=str(self.IP.text())
        camport=str(self.Port.text())
        username=str(self.Username.text())
        userpassword=str(self.Password.text())
        Database.getInstance().updataCaminfo(camid=camid,
                                             coordinate_x=float(coordinate_x),
                                             coordinate_y=float(coordinate_y),
                                             rate=float(rate),
                                             camip=camip,
                                             camport=camport,
                                             username=username,
                                             userpassword=userpassword)
        Database.getInstance().updataOpenflag(camid,1)
    
    # def closeEvent(self,QCloseEvent):
    #     res = QMessageBox.question(self,'提示','是否保存设置？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
    #     if res == QMessageBox.Yes:
    #         camid=str(self.ID.text())
    #         coordinate_x=str(self.X.text())
    #         coordinate_y=str(self.Y.text())
    #         rate=str(self.rate.text())
    #         camip=str(self.IP.text())
    #         camport=str(self.Port.text())
    #         username=str(self.Username.text())
    #         userpassword=str(self.Password.text())
    #         Database.getInstance().updataCaminfo(camid=camid,
    #                                             coordinate_x=float(coordinate_x),
    #                                             coordinate_y=float(coordinate_y),
    #                                             rate=float(rate),
    #                                             camip=camip,
    #                                             camport=camport,
    #                                             username=username,
    #                                             userpassword=userpassword)
    #         Database.getInstance().updataOpenflag(camid,1)
    #         QCloseEvent.accept()
    #     else:
    #         QCloseEvent.accept()

    # def addToDatabase(self):
    #     id = str(self.id.text())
    #     group = str(self.group.text())
    #     location = str(self.location.text())
    #     x = str(self.x_coord.text())
    #     y = str(self.y_coord.text())
    #     file = str(self.file.text())
    #     Database.getInstance().insertIntoCamera(id, location, x, y, group, file)
    #     self.destroy()

