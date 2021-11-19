from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from ui import Ui_MainWindow
import cv2
import numpy as np
import matplotlib.pyplot as plt




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.img = 0
        self.G_img = 0
        self.sobelX_img = 0
        self.sobelY_img = 0
        self.tsfm_img = 0


    def setup_control(self):
        self.ui.load_button.clicked.connect(self.__Load)
        self.ui.color_seperation_button.clicked.connect(self.Seperate)
        self.ui.color_tranform_button.clicked.connect(self.Color_transform)
        self.ui.blend_button.clicked.connect(self.Blend)
        self.ui.Gaussian_blur_button.clicked.connect(self.Gaussian1)
        self.ui.Bilateral_button.clicked.connect(self.Bilateral)
        self.ui.Median_button.clicked.connect(self.Median)
        self.ui.Gaussian_blur_button2.clicked.connect(self.Own_Gaussian)
        self.ui.sobelX_button.clicked.connect(self.SobelX)
        self.ui.sobelY_button.clicked.connect(self.SobelY)
        self.ui.magnitude_button.clicked.connect(self.Magnitude)
        self.ui.resize_button.clicked.connect(self.Resize)
        self.ui.translation_button.clicked.connect(self.Translate)
        self.ui.Rotate_scale_button.clicked.connect(self.Rotate)
        self.ui.shearing_button.clicked.connect(self.Shearing)
        
        self.warnMsg = QMessageBox()


    def __Load(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, "Select Image", "./", "*.jpg;*.png")
        if fileName:
            self.img = cv2.imread(str(fileName)) 
            cv2.imshow("1", self.img)
            print("Height : " + str(self.img.shape[0]))
            print("Width : " + str(self.img.shape[1]))


    def Seperate(self):
        if type(self.img) == type(0):
            self.warnMsg.setText("Need choose one image at least!")
            self.warnMsg.show()
        else:
            plt.figure(1)
            plt.title('Seperate Channel')
            zeros = np.zeros(self.img.shape[:2], dtype = "uint8")
            b, g, r = cv2.split(self.img)
            
            rimg = cv2.merge([r, zeros, zeros])
            gimg = cv2.merge([zeros, g, zeros])
            bimg = cv2.merge([zeros, zeros, b])

            plt.subplot(1, 3, 1)
            plt.title('red channel')
            plt.imshow(rimg)

            plt.subplot(1, 3, 2)
            plt.title('green channel')
            plt.imshow(gimg)

            plt.subplot(1, 3, 3)
            plt.title('blue channel')
            plt.imshow(bimg)

            plt.show()

    
    def Color_transform(self):
        if type(self.img) == type(0):
            self.warnMsg.setText("Need choose one image at least!")
            self.warnMsg.show()
        else:
            cv_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            mean_gray = np.sum(self.img, axis=2) / 3
            plt.figure(2)

            plt.subplot(1, 2, 1)
            plt.title('opencv gray')
            plt.imshow(cv_gray, cmap = "gray")

            plt.subplot(1, 2, 2)
            plt.title('mean gray')
            plt.imshow(mean_gray, cmap = "gray")

            plt.show()


    def __nothing(self, x):
        pass

    def Blend(self):
        if type(self.img) == type(0):
            self.warnMsg.setText("Need choose one image first")
            self.warnMsg.show()
        else:
            blend = 0
            fileName, fileType = QFileDialog.getOpenFileName(self, "Choose other Image", "./", "*.jpg;*.png")
            if fileName:
                blend_img = cv2.imread(str(fileName))
                self.warnMsg.setText("Use ESC to close window later")
                self.warnMsg.show()
                cv2.namedWindow('Blending')
                cv2.createTrackbar('Blend', 'Blending', 0, 255, self.__nothing)
                while(True):
                    blend = cv2.getTrackbarPos('Blend', 'Blending')
                    img_add = cv2.addWeighted(self.img, 1 - (blend / 255 ), blend_img, blend / 255, 0)
                    cv2.imshow('Blending', img_add)
                    if cv2.waitKey(1) == 27:
                        break
                cv2.destroyAllWindows()

    
    def Gaussian1(self):
        if type(self.img) == type(0):
            self.warnMsg.setText("Need choose one image first!")
            self.warnMsg.show()
        else:
            img_blur = cv2.GaussianBlur(self.img, (5, 5), 0 , cv2.BORDER_CONSTANT)
            cv2.namedWindow("blur image")
            cv2.imshow("blur image", img_blur)

    def Bilateral(self):
        if type(self.img) == type(0):
            self.warnMsg.setText("Need choose one image first!")
            self.warnMsg.show()
        else:
            img_blur = cv2.bilateralFilter(self.img, 9, 90, 90)
            cv2.namedWindow("blur image")
            cv2.imshow("blur image", img_blur)


    def Median(self):
        if type(self.img) == type(0):
            self.warnMsg.setText("Need choose one image first!")
            self.warnMsg.show()
        else:
           img_blur = np.hstack([
                cv2.medianBlur(self.img, 3),
                cv2.medianBlur(self.img, 5)
               ])
           cv2.namedWindow('Median Filter 3x3 & 5x5')
           cv2.imshow('Median Filter 3x3 & 5x5', img_blur)


    def __zero_padding(self, img, pLen):
        H = img.shape[0]
        W = img.shape[1]
        result = np.zeros([H + pLen * 2 ,W + pLen * 2], dtype = np.uint8)   
        result[pLen: pLen + H, pLen: pLen + W] = img.copy()
        return result

    def __convolution(self, img, kernel):   
        m, n = kernel.shape
        if(m == n ):
            #new_image = np.zeros((img.shape[0] - 2, img.shape[1] - 2 ), dtype = np.uint8)   # 'img' is after padding, so need to minus 2
            new_image = np.zeros((img.shape[0] - 2, img.shape[1] - 2 ))
            for i in range(img.shape[0] - 2):
                for j in range(img.shape[1] - 2):
                    tmp = abs( np.sum(img[i : i + m , j : j + m] * kernel) ) 
                    if tmp > 255:
                        tmp = 255
                    new_image[i][j] = tmp
            return new_image
      

    def Own_Gaussian(self):
        if type(self.img) == type(0):
            self.warnMsg.setText("Need to choose one image first!")
            self.warnMsg.show()

        else:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            pad_img = self.__zero_padding(gray, 1)
       
            Gaussian = np.array([[0.045, 0.122, 0.045],
                                [0.122, 0.332, 0.122],
                                [0.045, 0.122, 0.045]])
            #print(Gaussian.shape)
            #print(pad_img.shape)
            self.G_img = self.__convolution(pad_img, Gaussian)
            result = np.array(self.G_img, dtype = np.uint8)
            cv2.namedWindow("Gaussian Blur")
            cv2.imshow("Gaussian Blur", result)
            

    def SobelX(self):
        if type(self.G_img) == type(0):
            self.warnMsg.setText("You need to generate Gaussian blured image by 3.1 first !")
            self.warnMsg.show()

        else:
            pad_img = self.__zero_padding(self.G_img, 1)
            sobelX = np.array([(-1.0, 0.0, 1.0),
                               (-2.0, 0.0, 2.0),
                               (-1.0, 0.0, 1.0)])
            self.sobelX_img = self.__convolution(pad_img, sobelX)
            #result = result / np.linalg.norm(result) * 255
            result = np.array(self.sobelX_img, dtype = np.uint8)
            cv2.namedWindow("SobelX")
            cv2.imshow("SobelX", result)

    def SobelY(self):
        if type(self.G_img) == type(0):
            self.warnMsg.setText("You need to generate Gaussian blured image by 3.1 first !")
            self.warnMsg.show()

        else:
            pad_img = self.__zero_padding(self.G_img, 1)
            sobelX = np.array([(1.0, 2.0, 1.0),
                               (0.0, 0.0, 0.0),
                               (-1.0, -2.0, -1.0)])
            self.sobelY_img = self.__convolution(pad_img, sobelX)
            #result = result / np.linalg.norm(result) * 255
            result = np.array(self.sobelY_img, dtype = np.uint8)
            cv2.namedWindow("SobelY")
            cv2.imshow("SobelY", result)


    def Magnitude(self):
        if type(self.sobelX_img) == type(0) or type(self.sobelY_img) == type(0):
            self.warnMsg.setText("You need to generate sobelX & sobelY images first !")
            self.warnMsg.show()

        else:
            result = np.power(np.power(self.sobelX_img, 2) + np.power(self.sobelY_img, 2), 0.5)
            mins = np.min(result)
            maxs = np.max(result)
            result = (result - mins) / (maxs - mins ) * 255
            
            image = np.array(result, dtype = np.uint8)
            cv2.imshow("result", image)
           
    def Resize(self):
        if type(self.img) == type(0):
            self.warnMsg.setText("Need to choose one image first!")
            self.warnMsg.show()

        else:
            self.tsfm_img = cv2.resize(self.img, (256, 256), cv2.INTER_CUBIC)
            cv2.namedWindow("img")
            cv2.imshow('img', self.tsfm_img)
            

    def __get_translation_Matrix(self, x, y):
        return np.float32([[1, 0, x],
                          [0, 1, y]])
    
    def Translate(self):
        if type(self.tsfm_img) == type(0):
            self.warnMsg.setText("Need to resize first!")
            self.warnMsg.show()
        else:
           Matrix = self.__get_translation_Matrix(0, 60)
           self.tsfm_img = cv2.warpAffine(self.tsfm_img, Matrix, (self.tsfm_img.shape[1] + 60, self.tsfm_img.shape[0] + 60))
           cv2.namedWindow('img2')
           cv2.imshow('img2', self.tsfm_img)


    def Rotate(self):
        if type(self.tsfm_img) == type(0):
            self.warnMsg.setText("Need to resize first!")
            self.warnMsg.show()
        else:
            windowSize = (400, 300)
            imgCenter = (128, 188)
            Matrix = cv2.getRotationMatrix2D(imgCenter, 10, 0.5)
            self.tsfm_img = cv2.warpAffine(self.tsfm_img, Matrix, windowSize)
            cv2.namedWindow('img3')
            cv2.imshow('img3', self.tsfm_img)


    def Shearing(self):
        if type(self.tsfm_img) == type(0):
            self.warnMsg.setText("Need to resize first!")
            self.warnMsg.show()
        else:
            srcPoint = np.float32([[50, 50],
                                [200, 50],
                                [50,200]])
            dstPoint = np.float32([[10, 100],
                                [200, 50],
                                [100, 250]])

            Matrix = cv2.getAffineTransform(srcPoint, dstPoint)
            self.tsfm_img = cv2.warpAffine(self.tsfm_img, Matrix, (self.tsfm_img.shape[1], self.tsfm_img.shape[0]))
            cv2.namedWindow('img4')
            cv2.imshow('img4', self.tsfm_img)