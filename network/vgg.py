from torch import nn 
import torch
from torch import device
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms, models
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device : ', device)

batch_size = 32
trainSplit = 0.75
valSplit = 1 - trainSplit

transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root = '.\data', train = True, download = False, transform = transform)
testset = torchvision.datasets.CIFAR10(root = '.\data', train = False, download = False, transform = transform)

numTrainData = int(len(trainset) * trainSplit)
numValData = int(len(trainset) * valSplit)

trainset, valset = random_split(trainset, [numTrainData, numValData], generator = torch.Generator().manual_seed(42))

trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
valiloader = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle = False)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',  'dog', 'frog', 'horse', 'ship', 'truck')


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


H = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": []
    }

def Train():
    # hyperparameter
    epochs = 50
    learning_rate = 0.0001
    
    # network
    net = VGG16(num_classes = 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    
    
    for epoch in range(epochs):
        
        train_loss = []
        train_accs = []
        # train
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # init gradient
            optimizer.zero_grad()

            outputs = net(inputs)    # output : [batchsize, 10]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct = (outputs.argmax(1) == labels).float().mean().cpu().detach().numpy()
            train_loss.append(loss.cpu().detach().numpy())
            train_accs.append(correct)

            # print loss every 20 mini batch
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {sum(train_loss) / len(train_loss)}, Accuracy: {sum(train_accs) / len(train_accs) * 100}")
        H['train_loss'].append(sum(train_loss) / len(train_loss))
        H['train_acc'].append(sum(train_accs) / len(train_accs) * 100)

        #validation
        with torch.no_grad():
            net.eval()
            preds = []
            for inputs, labels in valiloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predict = torch.max(outputs.data, 1)
                correct = (predict == labels).float().mean().cpu().detach().numpy()
                preds.append(correct)

        H['test_acc'].append(sum(preds) / len(preds) * 100)


    print('Finish train')
    
    torch.save(net.state_dict(), './model.pt')


def Test():
    total_correct = 0
    total_image = 0
    test_accs = []
    with torch.no_grad():
        net.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predict = torch.max(outputs.data, 1)

            correct = (predict == labels).float().mean().cpu().detach().numpy()
            test_accs.append(correct)
    model_accuracy = sum(test_accs) / len(test_accs) * 100
   
    print(f'Model test accuracy: {model_accuracy}')


Train()
net = VGG16(num_classes=10).to(device)
net.load_state_dict(torch.load('./model.pt'))
print(net)
Test()

plt.style.use("ggplot")
plt.figure(1)
plt.plot(H['train_acc'], label = "train_acc")
plt.plot(H['test_acc'], label = "validation_acc")
plt.title("Train & Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("%")
plt.legend(loc="lower left")
plt.show()


plt.style.use("ggplot")
plt.figure(2)
plt.plot(H['train_loss'], label = "train_loss")
plt.title("Train Loss")
plt.xlabel("Epoch #")
plt.ylabel("loss")
plt.legend(loc="lower left")
plt.show()
