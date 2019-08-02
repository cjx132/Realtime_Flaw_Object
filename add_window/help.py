from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
#from Database import Database


class HelpForm(QMainWindow):
    def __init__(self, parent=None, ui=None):
        super(HelpForm, self).__init__(parent)
        loadUi("UI/help.ui", self)
    # def addToDatabase(self):
    #     id = str(self.id.text())
    #     group = str(self.group.text())
    #     location = str(self.location.text())
    #     x = str(self.x_coord.text())
    #     y = str(self.y_coord.text())
    #     file = str(self.file.text())
    #     Database.getInstance().insertIntoCamera(id, location, x, y, group, file)
    #     self.destroy()
