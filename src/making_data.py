#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:05:00 2018

@author: nirmal
"""
import os 
import csv
from itertools import izip

path_f='/media/nirmal/data/masters/sem2/computer vision/project 2/output/training'

path_f_test='/media/nirmal/data/masters/sem2/computer vision/project 2/output/test data faces'



filearray_f=[]
label=[]

filearray_f_test=[]
label_test=[]

for name in os.listdir(path_f):
    c=int(filter(str.isdigit,name))
    print(c)
    if c<10001:
        filearray_f.append(name)
        label.append(1)
        
    else:
        filearray_f.append(name)
        label.append(0)



for name in os.listdir(path_f_test):
    c=int(filter(str.isdigit, name))
    print(c)
    if c<1001:
        filearray_f_test.append(name)
        label_test.append(1)
        
    else:
        filearray_f_test.append(name)
        label_test.append(0)

with open('train.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(izip(filearray_f, label))

with open('test.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(izip(filearray_f_test, label_test))
