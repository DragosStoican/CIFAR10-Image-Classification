import sys
import os
import time
import torch
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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 500
BATCH_SIZE = 64
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(      
            nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3, padding=1), nn.BatchNorm2d(num_features=32), nn.relu(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=32), nn.relu(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=32), nn.relu(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=64), nn.relu(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=64), nn.relu(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=64), nn.relu(), 
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=128), nn.relu(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=128), nn.relu(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=128), nn.relu(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.4),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=256), nn.relu(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=256), nn.relu(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), nn.BatchNorm2d(num_features=256), nn.relu(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=256 * 2 * 2, out_features=128), nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.features(x)
        return x

def testModel(model, testLoader):
    correct = total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomChoice([
        transforms.RandomAffine(0, (0.2, 0.2)),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
])

trainset = torchvision.datasets.CIFAR10(root=f"{sys.path[0]}/data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=f"{sys.path[0]}/data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

net = Net().to(DEVICE)

trainAccuracyList = [0]
testAccuracyList = [0]
epochs = [0]
fig = plt.figure()
ax = plt.subplot()
plt.ylim([0, 100])

if args.r:
    try: os.remove(PATH)
    except: pass

def adjustLearningRate(optimizer):
    lr = 1e-3 * 0.993
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if os.path.isfile(PATH):
    net.load_state_dict(torch.load(PATH))
    print(f"Loaded state from: {PATH}")
if args.t or not os.path.isfile(PATH):
    trainingStart = time.time()
    bestAccuracy = 0
    print("No load state found, beginning training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = optim.Adam(net.parameters())
    for epoch in range(MAX_EPOCHS):
        start = time.time()
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        accuracy = round(100 * sum([testModel(net, testloader) for _ in range(1)]), 2)
        trainAccuracy = round(100 * sum([testModel(net, trainloader) for _ in range(1)]), 2)

        trainAccuracyList.append(trainAccuracy)
        testAccuracyList.append(accuracy)
        epochs.append(epoch+1)
        ax.plot(epochs, trainAccuracyList, color="blue")
        ax.plot(epochs, testAccuracyList, color="orange")
        fig.savefig('plot.png')

        print(f"Epoch {epoch+1} completed in {round(time.time() - start, 2)} seconds")
        print(f"Train accuracy: {trainAccuracy}% Test Accuracy: {accuracy}%")
        adjustLearningRate(optimizer)

        if (accuracy > bestAccuracy):
            torch.save(net.state_dict(), PATH)
            bestAccuracy = accuracy      
            print(f"New best: {bestAccuracy}")      
        
    print(f"\nFinished Training in {round(time.time() - trainingStart, 2)} seconds\n")


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
    
