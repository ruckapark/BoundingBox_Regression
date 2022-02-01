# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:17:51 2022

@author: George
"""

import os
import shutil
import numpy as np

#transfer into TrainSet and ValidationSet
good_images_dir = r'C:\Users\George\Documents\Python Scripts\DeepTest\train_good\dataset'
bad_images_dir = r'C:\Users\George\Documents\Python Scripts\DeepTest\train_bad\dataset'
train_dir = r'C:\Users\George\Documents\Python Scripts\DeepTest\Radix_NNdata\TrainSet'
valid_dir = r'C:\Users\George\Documents\Python Scripts\DeepTest\Radix_NNdata\ValidationSet'

#write 70% from each dataset into the train set, other 30 to the test set
train_ratio = 0.7
index_good,index_bad = np.arange(len(os.listdir(good_images_dir))//2),np.arange(len(os.listdir(bad_images_dir))//2)
index_good_train = np.sort(np.random.choice(index_good,int(train_ratio*len(index_good)),replace = False))
index_good_validation = np.delete(index_good,index_good_train)
index_bad_train = np.sort(np.random.choice(index_bad,int(train_ratio*len(index_bad)),replace = False))
index_bad_validation = np.delete(index_bad,index_bad_train)

#write images to parent directories
for i in index_good_train:
    #copy with shutil to other directory
    shutil.copy(r'{}\train_good_{}.jpg'.format(good_images_dir,i),r'{}\train_good_{}.jpg'.format(train_dir,i))
    shutil.copy(r'{}\train_good_{}.txt'.format(good_images_dir,i),r'{}\train_good_{}.txt'.format(train_dir,i))
    
for i in index_bad_train:
    #copy with shutil to other directory
    shutil.copy(r'{}\train_bad_{}.jpg'.format(bad_images_dir,i),r'{}\train_bad_{}.jpg'.format(train_dir,i))
    shutil.copy(r'{}\train_bad_{}.txt'.format(bad_images_dir,i),r'{}\train_bad_{}.txt'.format(train_dir,i))

for i in index_good_validation:
    #copy with shutil to other directory
    shutil.copy(r'{}\train_good_{}.jpg'.format(good_images_dir,i),r'{}\train_good_{}.jpg'.format(valid_dir,i))
    shutil.copy(r'{}\train_good_{}.txt'.format(good_images_dir,i),r'{}\train_good_{}.txt'.format(valid_dir,i))
    
for i in index_bad_validation:
    #copy with shutil to other directory
    shutil.copy(r'{}\train_bad_{}.jpg'.format(bad_images_dir,i),r'{}\train_bad_{}.jpg'.format(valid_dir,i))
    shutil.copy(r'{}\train_bad_{}.txt'.format(bad_images_dir,i),r'{}\train_bad_{}.txt'.format(valid_dir,i))
    
#%% write as data and put bbs in csv file
ims = [f for f in os.listdir(train_dir) if 'jpg' in f]
order = np.arange(len(ims))
np.random.shuffle(order)
for i,x in enumerate(order):
    os.rename(r'{}\{}'.format(train_dir,ims[x]),r'{}\data{}.jpg'.format(train_dir,i))
    os.rename(r'{}\{}.txt'.format(train_dir,ims[x].split('.')[0]),r'{}\data{}.txt'.format(train_dir,i))
    
ims = [f for f in os.listdir(valid_dir) if 'jpg' in f]
order = np.arange(len(ims))
np.random.shuffle(order)
for i,x in enumerate(order):
    os.rename(r'{}\{}'.format(valid_dir,ims[x]),r'{}\data{}.jpg'.format(valid_dir,i))
    os.rename(r'{}\{}.txt'.format(valid_dir,ims[x].split('.')[0]),r'{}\data{}.txt'.format(valid_dir,i))