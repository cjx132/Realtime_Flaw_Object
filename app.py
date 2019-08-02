import sys
# import qdarkstyle
from PyQt5.QtWidgets import QApplication,QSplashScreen
from MainWindow import MainWindow
from PyQt5 import QtGui,QtCore

def main():
    app = QApplication(sys.argv)
    ash = QSplashScreen(QtGui.QPixmap("UI/NUI.png"))
    ash.setFont(QtGui.QFont('Microsoft YaHei UI',20))
    ash.show()
    ash.showMessage("系统启动中,请稍候",QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom, QtCore.Qt.green)
    main_window = MainWindow()
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main_window.show()
    ash.finish(main_window)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
