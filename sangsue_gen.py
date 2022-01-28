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
import matplotlib.pyplot as plt

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

from scipy.interpolate import interp1d,splev,splrep,splprep

if __name__ == "__main__":
    
    # create numpy array of 164*164 (light grey colour)
    grey = np.random.randint(150,225)/255
    img = grey*np.ones((328,328))
    
    # create black line of length 15-40
    black = np.random.randint(0,75)/255
    width = np.random.randint(2,6)
    
    #bb dimensions
    start_x = np.random.randint(20,200)
    start_y = np.random.randint(20,200)
    b_height,b_width = np.random.randint(10,120),np.random.randint(10,120)
    bb = np.array([start_x,start_y,b_height,b_width])
    
    def dist_euc(x1,y1,x2,y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    def gen_tail(x,y,w,h,x1,y1):
        x2,y2 = np.random.randint(x,x + w),np.random.randint(y,y + h)
        if dist_euc(x1,y1,x2,y2) > 4:
            return x2,y2
        else :
            return gen_tail(x,y,w,h,x1,y1)
    
    def gen_midpoint(x,y,w,h,x1,y1,x2,y2):
        midx,midy = np.random.randint(x,x + w),np.random.randint(y,y + h)
        if (dist_euc(midx,midy,x1,y1) > 4) & (dist_euc(midx,midy,x2,y2) > 4):
            return midx,midy
        else: 
            return gen_midpoint(x,y,w,h,x1,y1,x2,y2)
        
    
    #define head tail and midpoint
    head_x,head_y = np.random.randint(start_x,start_x + b_width),np.random.randint(start_y,start_y + b_height)
    tail_x,tail_y = gen_tail(start_x,start_y,b_width,b_height,head_x,head_y)
    mid_x,mid_y = gen_midpoint(start_x,start_y,b_width,b_height,head_x,head_y,tail_x,tail_y)
    
    #draw three points on image
    points = [[head_x,head_y],[mid_x,mid_y],[tail_x,tail_y]]
    for point in points:
        cv2.circle(img,tuple(point),2,black,thickness = -1)
        
    cv2.imshow('test',img)
    
    #fit spline
    x,y = [head_x,mid_x,tail_x],[head_y,mid_y,tail_y]
    f = interp1d(x, y,kind = 'quadratic')
    x_int = np.linspace(np.min(x),np.max(x))
    y_int = f(x_int)
    
    plt.plot(x_int,y_int)
    
    """
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
    
    uncomment to check location of bb
    cv2.rectangle(img,(start_x-2,start_y-2),(end_x+2,end_y+2),1,1)
    
    check flip
    cv2.rectangle(flip,(bb_flip[0]-2,bb_flip[1]-2),(bb_flip[0]+bb_flip[2]+2,bb_flip[1]+bb_flip[3]+2),1,1)
    cv2.imshow('test',flip)
    """