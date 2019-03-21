import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
import math
import collections
import sys

from resnet32 import *
from resnet56 import *
from resnet101 import *
from pyramidnet import *
from densenet import *
from student import *

def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

"""Train model without teacher"""
def train_model_naive(model, train_loader, test_loader, num_epochs, optimizer, scheduler, criterion):
    epoch_loss = []
    for epoch in range(num_epochs):
        batch_loss = []
        for i, (images, labels) in enumerate(train_loader):

            images = images.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        cur_epoch_loss = sum(batch_loss) / len(batch_loss)
        print('Epoch:', epoch, 'Loss:', str(cur_epoch_loss))
        epoch_loss.append(cur_epoch_loss)
        scheduler.step(cur_epoch_loss)
        
        if epoch % 10 == 9:
            test_model(model, test_loader)
    
    return epoch_loss

"""Train model with teacher"""
def train_model_teacher(student_model, teacher_model, train_loader, test_loader, num_epochs, optimizer, scheduler, criterion):
    epoch_loss = []
    for epoch in range(num_epochs):
        batch_loss = []
        for i, (images, labels) in enumerate(train_loader):

            images = images.cuda()
            
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            
            optimizer.zero_grad()
            student_outputs = student_model(images)
            loss = criterion(student_outputs, teacher_outputs.detach())
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        cur_epoch_loss = sum(batch_loss) / len(batch_loss)
        print('Epoch:', epoch, 'Loss:', str(cur_epoch_loss))
        epoch_loss.append(cur_epoch_loss)
        scheduler.step(cur_epoch_loss)
        
        if epoch % 10 == 9:
            test_model(student_model, test_loader)
#             test_model(teacher_model, test_loader)
    return epoch_loss

"""Test model"""
def test_model(model, test_loader):
    # Test the model
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(c.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    print('total acc:', sum(class_correct) / sum(class_total))


if __name__ == '__main__':
    batch_size = 64
    test_batch_size = 64

    train_loader, _ = cifar_loaders(batch_size)
    _, test_loader = cifar_loaders(test_batch_size)

    #load resnet32 weight
    res32 = resnet32().cuda()
    checkpoint = torch.load('resnet32.pth')
    res32.load_state_dict(checkpoint['net'])
    res32.eval()

    #load resnet56 weight
    res56 = resnet56().cuda()
    res56.load_state_dict(torch.load('resnet56.pth'))
    res56.eval()

    #load resnet101 weight
    res101 = ResNet101().cuda()
    res101.load_state_dict(torch.load('resnet101.pth'))
    res101.eval()

    #load Densenet weight
    densenet = densenet_cifar().cuda()
    checkpoint = torch.load('densenet.ckpt')
    checkpoint2 = collections.OrderedDict()
    for i in checkpoint['net']:
        checkpoint2[i[7:]] = checkpoint['net'][i]
    densenet.load_state_dict(checkpoint2)
    densenet.eval()

    #load Pyramidnet
    pyramidnet =  pyramidnet().cuda()
    checkpoint = torch.load('PyramidNet.pth')
    pyramidnet.load_state_dict(checkpoint['net'])
    pyramidnet.eval()

    #set student model
    CNN5LAYER = simplecnn().cuda()
    CNN7LAYER = simplecnn2().cuda()
    MLPLAYER  = simplemlp().cuda()
    CNN4LAYER_wider = simplecnn3().cuda()
    CNN4LAYER_narrower = simplecnn1().cuda()


    # Hyperparameters
    #################### Hyperparameters ####################
    num_epochs = 500
    learning_rate = 0.001
    factor = 0.5
    patience = 10
    weight_decay = 0
    criterion = nn.CrossEntropyLoss()
    #################### Hyperparameters ####################
    

    ### Example of running one sample
    ### use CNN7LAYER as student and densenet as teacher
    student_net = CNN7LAYER
    teacher_net = densenet
    ### set optimizer and 
    optimizer = torch.optim.Adam(student_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', 
                                                        factor=factor,
                                                        patience=patience, 
                                                        verbose=True, 
                                                        threshold=1e-3)

    ### train the student model without help of teacher to see the performance
    train_model_naive(student_net, train_loader, test_loader, num_epochs,optimizer = optimizer, scheduler = scheduler, criterion = criterion)
    test_model(student_net,test_loader)
    ### use teacher to help train the student
    train_model_teacher(student_net, teacher_net, train_loader, test_loader, num_epochs, optimizer = optimizer, scheduler = scheduler, criterion = criterion)
    test_model(student_net,test_loader)