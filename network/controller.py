from PyQt5 import QtWidgets, QtGui, QtCore
from ui import Ui_MainWindow
from PyQt5.QtWidgets import QMessageBox
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch import nn 
import torch
from torch import device
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms, models
import torchvision
from torchsummary import summary
import random


classes = ['plane', 'automobile', 'bird', 'cat', 'deer',  'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG16(nn.Module):
    def __init__(self, num_classes = 100):
        super().__init__()
        net = models.vgg16(pretrained = False)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()


    def setup_control(self):
        self.ui.ShowImage.clicked.connect(self.TrainImage)
        self.ui.ShowPara.clicked.connect(self.Parameter)
        self.ui.ShowModel.clicked.connect(self.Shorcut)
        self.ui.ShowAcc.clicked.connect(self.Acc)
        self.ui.Test.clicked.connect(self.Test)
        self.warnMsg = QMessageBox()        


    def TrainImage(self):
        trainset = torchvision.datasets.CIFAR10(root = '.\data', train = False, download = False, transform = transforms.ToTensor())
        
        images = []
        labels = []
        #print(len(trainset))
        rand = [random.randint(1, 10000) for i in range(10)]

        for i in rand:
            (data, label) = trainset[i]
            labels.append(label)
            result = data.cpu().numpy()
            result = np.transpose(result, (1, 2, 0))
            images.append(result)
            #result = transforms.ToPILImage()(data).resize((200, 200))
            #result.show()
        
        plt.close()
        fig = plt.figure(1)
        for i in range(9):
            fig.add_subplot(3, 3, i + 1)
            plt.title(classes[labels[i]])
            plt.axis('off')
            plt.imshow(images[i])
        plt.show()
        

    def Parameter(self):
        print(f"HyperParameters: \n batch size: 32 \n learning rate: 0.0001 \n optimizer: Adam")
    

    def Shorcut(self):
        #Train()
        net = VGG16(num_classes=10).to(device)
        net.load_state_dict(torch.load('./model.pt'))
        #print(summary(net, torch.zeros((32, 3, 224, 224)).to(device), show_input=False) )
        print("Use Summary")
        summary(net, (3, 32, 32), device = 'cuda')
        print("\n")
        print("Use Print")
        print(net)

    def Acc(self):
        acc = cv2.imread("./Figure_1.png")
        loss = cv2.imread("./Figure_2.png")
        cv2.imshow("Accuracy", acc)
        cv2.imshow("Loss", loss)

    
    def Test(self):
        batchSize = 32
        txt = self.ui.lineEdit.text()
        num = 0
        try:
            num = int(txt)
        except :
            self.warnMsg.setText("You need input number or it else defualt to 0")
            self.warnMsg.show()
            self.warnMsg.exec()
            num = 0
            
        transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR10(root = '.\data', train = False, download = False, transform = transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = False)
        showset = torchvision.datasets.CIFAR10(root = '.\data', train = False, download = False, transform = transforms.ToTensor())


        net = VGG16(num_classes=10).to(device)
        net.load_state_dict(torch.load('./model.pt'))
        net.eval()
        num = num % len(testset)
        pltx = np.arange(len(classes))
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                if i == num:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    outputs = outputs.reshape(-1)
                    outputs = nn.functional.softmax(outputs, dim = 0)
                    outputs = outputs.cpu().detach().numpy()

                    #outputs = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs))

                    print('outputs = ', outputs)

                    img, _ = showset[i]
                    img = img.cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(img)
                    plt.subplot(1, 2, 2)
                    plt.bar(pltx, outputs, color = 'blue', width = 0.5)
                    plt.xticks(pltx , classes, rotation = 30)
                    plt.xlabel('Classes')
                    plt.ylabel('Outputs')
                    break
                    
            plt.show()

        print("finish")