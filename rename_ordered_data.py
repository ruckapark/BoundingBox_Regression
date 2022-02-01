# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:10:04 2022

@author: George
"""
import os

root = r'C:\Users\George\Documents\Python Scripts\DeepTest\train_good\dataset'
os.chdir(root)
p = 'good'

files = [f for f in os.listdir() if 'jpg' in f]
indexes = sorted([int(f.split('.')[0][5:]) for f in files])
for i,x in enumerate(indexes):
    os.rename('train{}.jpg'.format(x),'train_{}_{}.jpg'.format(p,i))
    os.rename('train{}.txt'.format(x),'train_{}_{}.txt'.format(p,i))