# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950, 597)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
      
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(710, 40, 191, 441))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.resize_button = QtWidgets.QPushButton(self.groupBox_2)
        self.resize_button.setGeometry(QtCore.QRect(10, 40, 171, 31))
        self.resize_button.setObjectName("resize_button")
        self.translation_button = QtWidgets.QPushButton(self.groupBox_2)
        self.translation_button.setGeometry(QtCore.QRect(10, 130, 171, 31))
        self.translation_button.setObjectName("translation_button")
        self.Rotate_scale_button = QtWidgets.QPushButton(self.groupBox_2)
        self.Rotate_scale_button.setGeometry(QtCore.QRect(10, 250, 171, 31))
        self.Rotate_scale_button.setObjectName("Rotate_scale_button")
        self.shearing_button = QtWidgets.QPushButton(self.groupBox_2)
        self.shearing_button.setGeometry(QtCore.QRect(10, 370, 171, 31))
        self.shearing_button.setObjectName("shearing_button")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(480, 30, 191, 441))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.Gaussian_blur_button2 = QtWidgets.QPushButton(self.groupBox_3)
        self.Gaussian_blur_button2.setGeometry(QtCore.QRect(10, 40, 171, 31))
        self.Gaussian_blur_button2.setObjectName("Gaussian_blur_button2")
        self.sobelX_button = QtWidgets.QPushButton(self.groupBox_3)
        self.sobelX_button.setGeometry(QtCore.QRect(10, 130, 171, 31))
        self.sobelX_button.setObjectName("sobelX_button")
        self.sobelY_button = QtWidgets.QPushButton(self.groupBox_3)
        self.sobelY_button.setGeometry(QtCore.QRect(10, 240, 171, 31))
        self.sobelY_button.setObjectName("sobelY_button")
        self.magnitude_button = QtWidgets.QPushButton(self.groupBox_3)
        self.magnitude_button.setGeometry(QtCore.QRect(10, 360, 171, 31))
        self.magnitude_button.setObjectName("magnitude_button")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(20, 30, 191, 441))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.load_button = QtWidgets.QPushButton(self.groupBox_5)
        self.load_button.setGeometry(QtCore.QRect(10, 30, 171, 31))
        self.load_button.setObjectName("load_button")
        self.color_seperation_button = QtWidgets.QPushButton(self.groupBox_5)
        self.color_seperation_button.setGeometry(QtCore.QRect(10, 130, 171, 31))
        self.color_seperation_button.setObjectName("color_seperation_button")
        self.color_tranform_button = QtWidgets.QPushButton(self.groupBox_5)
        self.color_tranform_button.setGeometry(QtCore.QRect(10, 220, 171, 31))
        self.color_tranform_button.setObjectName("color_tranform_button")
        self.blend_button = QtWidgets.QPushButton(self.groupBox_5)
        self.blend_button.setGeometry(QtCore.QRect(10, 350, 171, 31))
        self.blend_button.setObjectName("blend_button")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(250, 30, 191, 441))
        font = QtGui.QFont()
        font.setFamily("Adobe Arabic")
        font.setPointSize(16)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.Gaussian_blur_button = QtWidgets.QPushButton(self.groupBox_4)
        self.Gaussian_blur_button.setGeometry(QtCore.QRect(10, 80, 171, 31))
        self.Gaussian_blur_button.setObjectName("Gaussian_blur_button")
        self.Bilateral_button = QtWidgets.QPushButton(self.groupBox_4)
        self.Bilateral_button.setGeometry(QtCore.QRect(10, 180, 171, 31))
        self.Bilateral_button.setObjectName("Bilateral_button")
        self.Median_button = QtWidgets.QPushButton(self.groupBox_4)
        self.Median_button.setGeometry(QtCore.QRect(10, 310, 171, 31))
        self.Median_button.setObjectName("Median_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1192, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        
        self.groupBox_2.setTitle(_translate("MainWindow", "Transformation"))
        self.resize_button.setText(_translate("MainWindow", "4.1 Resize"))
        self.translation_button.setText(_translate("MainWindow", "4.2 Translation"))
        self.Rotate_scale_button.setText(_translate("MainWindow", "4.3 Rotation, Scaling"))
        self.shearing_button.setText(_translate("MainWindow", "4.4 Shesring"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Edge Detection"))
        self.Gaussian_blur_button2.setText(_translate("MainWindow", "3.1 Gaussian Blur"))
        self.sobelX_button.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.sobelY_button.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.magnitude_button.setText(_translate("MainWindow", "3.4 Magnitude"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Image Processing"))
        self.load_button.setText(_translate("MainWindow", "1.1 Load Image"))
        self.color_seperation_button.setText(_translate("MainWindow", "1.2 Color Seperation"))
        self.color_tranform_button.setText(_translate("MainWindow", "1.3 Color Transfornation"))
        self.blend_button.setText(_translate("MainWindow", "1.4 Blending"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Image Smoothing"))
        self.Gaussian_blur_button.setText(_translate("MainWindow", "2.1 Gaussian Blur"))
        self.Bilateral_button.setText(_translate("MainWindow", "2.2 Blateral Filter"))
        self.Median_button.setText(_translate("MainWindow", "2.3 Median Filter"))