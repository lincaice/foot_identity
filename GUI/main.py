from food_identity_ui import *
from PyQt5.QtWidgets import QApplication,QMainWindow,QDialog,QWidget
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widgets = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(widgets)
    widgets.show()

    sys.exit(app.exec_())