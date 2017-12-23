# multi-class classifier of furniture
# 5-classes : ["bed","chair","sofa","swivelchair","table"]

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

plt.ion()

# the path to the directory containing the current file
dir_path = os.path.dirname(os.path.realpath(__file__))

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(200),
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

use_gpu = torch.cuda.is_available()

classes = ["bed","chair","sofa","swivelchair","table"]
num_of_classes = len(classes)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training phase
        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in trainloader:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()


            m = nn.LogSoftmax()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(m(outputs), labels)
            # forward
#            outputs = model(inputs)
#            _, preds = torch.max(outputs.data, 1)
#            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(trainset)
            epoch_acc = running_corrects / len(trainset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                "train", epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# transfer learning resnet18
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_of_classes)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.NLLLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# train model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)