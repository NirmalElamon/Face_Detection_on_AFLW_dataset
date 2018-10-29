#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:58:38 2018

@author: nirmal
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import Face
import Face_test

'''
STEP 1: LOADING DATASET
'''
csv_path='/media/nirmal/data/masters/sem2/computer vision/project 2/train.csv'
root_dir='/media/nirmal/data/masters/sem2/computer vision/project 2/output/training/'
transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])

train_dataset =Face.CustomDatasetFromImages(csv_path,root_dir,transformations)

root_dir1='/media/nirmal/data/masters/sem2/computer vision/project 2/output/test data faces/'
csv_path1='/media/nirmal/data/masters/sem2/computer vision/project 2/test.csv'
test_dataset =Face_test.CustomDatasetFromImages(csv_path1,root_dir1,transformations)
#transforms.Scale(32),

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)
#num_epochs=30
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1 (readout)
         
        self.fc1 = nn.Linear(6 * 5 * 5, 1)
        
        
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        #print(out.size())
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        #print(out.size())
        # Max pool 2 
        out = self.maxpool2(out)
       
        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 6*5*5)
        out = out.view(out.size(0), -1)
        #print(out.size())
        # Linear function (readout)
        out = self.fc1(out)
        
        
        return out
       
model = CNNModel()

#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model.cuda()
    
criterion = nn.MSELoss()

learning_rate = 0.001 

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #print("entered training")
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
      
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        outputs = model(images)
        outputs=outputs.cpu()
        labels=labels.cpu()
        
        #outputs=torch.LongTensor(outputs)
        #labels=torch.LongTensor(labels)
        labels=labels.type(torch.FloatTensor)
       
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs,labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        iter += 1
        
        if iter %500==0:
            # Calculate Accuracy 
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in train_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)
                
                # Forward pass only to get logits/output
                outputs = model(images)
                
                # Get predictions from the maximum value
                predicted=outputs
                predicted=predicted
                # Total number of labels
                total += labels.size(0)
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                # Total correct predictions
                y=labels.cpu()
                y=y.type(torch.FloatTensor)
                for i in range(len(predicted)):
                    
                    if predicted[i]>=0.5:
                        predicted[i]=1
                    else:
                        predicted[i]=0
                if torch.cuda.is_available():
                    x= (predicted.cpu() == y).sum()
                    correct+=x.data.cpu().numpy()
                else:
                    correct += (predicted == labels).sum()
                
            accuracy = 100 * correct / total
          
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))
