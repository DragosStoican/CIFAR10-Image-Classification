import sys
import os
import time
import torch
import pickle
import argparse
import matplotlib.pyplot as plt

import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store_true')
parser.add_argument('-r', action='store_true')
args = parser.parse_args()
PATH = f"{sys.path[0]}\\cifar_net.pth"
PROGRESS_DATA = f"{sys.path[0]}\\progress.bin"
PROGRESS_GRAPH = f"{sys.path[0]}\\plot.png"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 100
ERROR_TRAIN_EPOCH = 0
BATCH_SIZE = 64
TEST_REPEATS = 1
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = self.createLayers([(3, 64, 3, 0.2), (64, 128, 3, 0.3), (128, 256, 3, 0.4), (256, 512, 4, 0.5)])

    def forward(self, x):
        x = self.features(x)
        return x
    
    def createConvLayer(self, in_channels, out_channels, layers, dropout):
        items = []
        for _ in range(layers):
            items.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3, padding=1))
            items.append(nn.BatchNorm2d(num_features=out_channels))
            items.append(nn.ReLU())
            in_channels = out_channels
        items.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout != 0:    
            items.append(nn.Dropout(p=dropout))
        return items
    
    def createLayers(self, design):
        layers = []
        for param in design:
            layers.extend(self.createConvLayer(*param))
        layers.extend([
            nn.Flatten(),
            nn.Linear(in_features=512 * 2 * 2, out_features=128), nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=10)
        ])
        return nn.Sequential(*layers)


def testModel(model, testLoader, getErrors=False):
    totalCorrect = total = 0
    errors = []
    model.eval()
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct = (predicted == labels)
            totalCorrect += correct.sum().item()
            if getErrors and (correct.sum().item() != labels.size(0)):
                ims = [i for i,v in zip(images.tolist(), correct.tolist()) if not v]
                lbs = [l for l,v in zip(labels.tolist(), correct.tolist()) if not v]
                errors.append([
                    torch.tensor(ims),
                    torch.tensor(lbs)
                ])
    model.train()
    return (totalCorrect / total, errors) if getErrors else (totalCorrect / total)

def adjustLearningRate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.993

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomChoice([
        transforms.RandomAffine(0, (0.2, 0.2)),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3)
    ])
])

trainset = torchvision.datasets.CIFAR10(root=f"{sys.path[0]}/data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
errorLoader = [d for d in trainloader]

testset = torchvision.datasets.CIFAR10(root=f"{sys.path[0]}/data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

net = Net().to(DEVICE)

loader = trainloader
learningRate = 1e-3
trainAccuracyList = [0]
testAccuracyList = [0]
lossRates = [1]
epochs = [0]
fig = plt.figure()
ax = plt.subplot()
plt.ylim([0, 100])


if args.r:
    for f in [PATH, PROGRESS_DATA, PROGRESS_GRAPH]:
        try: os.remove(f)
        except: pass

if os.path.isfile(PATH):
    net.load_state_dict(torch.load(PATH))
    with open(PROGRESS_DATA, "rb") as binFile:
        learningRate = pickle.load(binFile)
        trainAccuracyList = pickle.load(binFile)
        testAccuracyList = pickle.load(binFile)
        lossRates = pickle.load(binFile)
        epochs = pickle.load(binFile)
        learningRate *= 0.993**epochs[-1]
    print(f"Loaded state from: {PATH}")
else:
    print("No load state. Creating new model.")
if args.t or not os.path.isfile(PATH):
    ax.plot(epochs, trainAccuracyList, color="blue", label="Train Accuracy (%)")
    ax.plot(epochs, testAccuracyList, color="orange", label="Test Accuracy (%)")
    ax.plot(epochs, lossRates, color="green", label="loss (100 * (1-loss))")
    plt.legend(loc="lower right")
    fig.savefig(PROGRESS_GRAPH, dpi=300)

    trainingStart = time.time()
    bestAccuracy = max(testAccuracyList)
    print("Beginning training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)

    for epoch in range(epochs[-1]+1, MAX_EPOCHS):
        if epoch > ERROR_TRAIN_EPOCH:
            loader = errorLoader
        start = time.time()
        for i, data in enumerate(loader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        accuracy = round((100 / TEST_REPEATS) * sum([testModel(net, testloader) for _ in range(TEST_REPEATS)]), 2)
        trainAccuracy, errorLoader = testModel(net, trainloader, getErrors=True)
        trainAccuracy = round(100 * trainAccuracy, 2)

        trainAccuracyList.append(trainAccuracy)
        testAccuracyList.append(accuracy)
        lossRates.append(100 * (1-loss.item()))
        epochs.append(epoch)
        ax.plot(epochs, trainAccuracyList, color="blue", label="Train Accuracy (%)")
        ax.plot(epochs, testAccuracyList, color="orange", label="Test Accuracy (%)")
        ax.plot(epochs, lossRates, color="green", label="loss")
        fig.savefig(PROGRESS_GRAPH, dpi=300)
        with open(PROGRESS_DATA, "wb") as binFile:
            pickle.dump(learningRate, binFile)
            pickle.dump(trainAccuracyList, binFile)
            pickle.dump(testAccuracyList, binFile)
            pickle.dump(lossRates, binFile)
            pickle.dump(epochs, binFile)

        print(f"Epoch {epoch} completed in {round(time.time() - start, 2)} seconds")
        print(f"Train accuracy: {trainAccuracy}% Test Accuracy: {accuracy}% Loss: {loss}")
        adjustLearningRate(optimizer)

        if (accuracy > bestAccuracy):
            torch.save(net.state_dict(), PATH)
            bestAccuracy = accuracy      
            print(f"New best: {bestAccuracy}")      
        
    print(f"\nFinished Training in {round(time.time() - trainingStart, 2)} seconds\n")

net.eval()
print(f"Accuracy of the network: {round(10 * sum([testModel(net, testloader) for _ in range(10)]), 2)}%\n")
classCorrect = [0 for _ in CLASSES]
classTotal = [0 for _ in CLASSES]
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = net(inputs)
        _, predictions = torch.max(outputs, 1)
        for prediction, label in zip(predictions, labels):
            classTotal[label] += 1
            classCorrect[label] += int(prediction.item() == label.item())

print("Object : Accuracy")
for i in range(10):
    print(" %5s : %2d %%" % (CLASSES[i], 100 * classCorrect[i] / classTotal[i]))
    
