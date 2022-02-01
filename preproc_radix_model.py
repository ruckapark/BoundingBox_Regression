# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:42:46 2022

Check format of bounding box and add to one file for NN training.

@author: George
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def write_boundingboxes():
    return None

def transform_bb(bb,shape = 128):
    """ transform to x,y in top left corner of bounding box """
    [x,y,w,h] = np.rint(shape*bb)
    return [x-(w//2),y-(h//2),w,h]

#change directory to dataset
dataset_dir = r'C:\Users\George\Documents\Python Scripts\DeepTest\Radix_NNdata\ValidationSet'
os.chdir(dataset_dir)
root = os.getcwd()

#files
ind = np.random.randint(0,1000)
test_file = 'data{}.jpg'.format(ind)
test_bb = 'data{}.txt'.format(ind)

#read image with cv2 - 0 grayscale
test_im = cv2.imread('data{}.jpg'.format(ind),0)


#read bounding box
test_bb = np.loadtxt(test_bb)[1:-1]
[x,y,w,h] = np.array(test_bb*128,dtype = np.uint)

cv2.rectangle(test_im,(x-(w//2),y-(h//2)),(x+(w//2),y+(h//2)),1,1)
cv2.imshow('test',test_im)

#run code to x,y top left and w,h with 128 resolution
bbs = []
files = [f for f in os.listdir() if 'txt' in f]
for i in range(len(files)):
    bb = np.loadtxt('data{}.txt'.format(i))[1:5]
    bb = transform_bb(bb)
    bbs.append(bb)
    
bbs = pd.DataFrame(np.array(bbs,dtype = np.uint))
bbs.to_csv(r'{}\bb.csv'.format(dataset_dir),header = False,index = False)