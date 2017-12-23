# multi-class classifier of furniture
# 5-classes : ["bed","chair","sofa","swivelchair","table"]

import torch
import torchvision
import os
from torchvision import transforms, datasets

# the path to the directory containing the current file
dir_path = os.path.dirname(os.path.realpath(__file__))

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(240),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

#data_transform = transforms.Compose(
#    [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.ImageFolder(root=dir_path+'/dataset/train',
                                           transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
print("trainloader ready!")

testset = datasets.ImageFolder(root=dir_path+'/dataset/test',
                                           transform=data_transform)

testloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
print("testloader ready!")

classes = ["bed","chair","sofa","swivelchair","table"]

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
## show images
#imshow(torchvision.utils.make_grid(images))
## print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#import torch.optim as optim
#
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
#for epoch in range(9):  # loop over the dataset multiple times
#
#    running_loss = 0.0
#    for i, data in enumerate(trainloader, 0):
#        # get the inputs
#        inputs, labels = data
#
#        # wrap them in Variable
#        inputs, labels = Variable(inputs), Variable(labels)
#
#        # zero the parameter gradients
#        optimizer.zero_grad()
#
#        # forward + backward + optimize
#        outputs = net(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#
#        # print statistics
#        running_loss += loss.data[0]
#        if i % 2000 == 1999:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0
#
#print('Finished Training')