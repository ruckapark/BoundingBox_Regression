# -*- coding: utf-8 -*-
"""
Create dataset for basic keras detection network

The same code could be used to create a test set for validation (basic dataaugmentation used with flip)
"""

import os
import cv2
import numpy as np
from random import shuffle
from csv import writer

def gen_img():

    # create numpy array of 164*164 (light grey colour)
    grey = np.random.randint(150,225)/255
    img = grey*np.ones((164,164))
    
    # create black line of length 15-40
    black = np.random.randint(0,75)/255
    width = np.random.randint(2,6)
    length = np.random.randint(15,40)
    start_x = np.random.randint(2,122)   #in image limits
    start_y = np.random.randint(2,122)   #in image limits
    orientation = np.random.randint(1,89)           #angle
    factor = np.pi/180
    
    end_x = start_x + int(length*np.cos(orientation*factor))
    end_y = start_y + int(length*np.sin(orientation*factor))
    
    # draw line
    cv2.line(img,(start_x,start_y),(end_x,end_y),black,width)
    
    # bounding box
    bb = np.array([start_x-1,start_y-1,end_x-start_x+1,end_y-start_y+1])
    
    """ uncomment to check location of bb
    cv2.rectangle(img,(start_x-2,start_y-2),(end_x+2,end_y+2),1,1)
    
    check flip
    cv2.rectangle(flip,(bb_flip[0]-2,bb_flip[1]-2),(bb_flip[0]+bb_flip[2]+2,bb_flip[1]+bb_flip[3]+2),1,1)
    cv2.imshow('test',flip)
    """
    
    return img,bb

if __name__ == "__main__":
    
    os.chdir('ValidationSet')
    dataset_size = 1000
    dataset = []
    for x in range(dataset_size):
        
        img,bb = gen_img()
        dataset.append([img,bb])
        
        
    for x in range(dataset_size):
        data = dataset[x]
        img = data[0]
        bb = data[1]
        
        flip = cv2.flip(img,1)
        bb_flip = np.copy(bb)
        bb_flip[0] = 164 - bb[0] - bb[2]
        
        dataset.append([flip,bb_flip])
        
    shuffle(dataset)
    
    for x,data in enumerate(dataset):
        
        
        cv2.imwrite("test_{}.png".format(x),255*data[0])
        with open('bb.csv','a+',newline = '') as write:
            csv_writer = writer(write)
            csv_writer.writerow(data[1])