# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:17:15 2019

@author: Daniel
modified by Joachim Axelsen, s144015, nov-19

"""

import cv2
import numpy as np
import time
import argparse
import glob
import os.path
from os import path
import progressbar

parser = argparse.ArgumentParser(description='Chop video file into frames.')
parser.add_argument('videopath', type=str, help='path to video file that is to be chopped')
parser.add_argument('savepath', type=str, help='path to the folder where to place the chopped frames')
parser.add_argument('--frameskip', type=int, help='integer denoting the number of frames to skip before a new frame is saved. Default=15.', default=15)
parser.add_argument('--rewrite', type=int, help='set to 1 if you want to rewrite the names of the images already in the folder', default=0)

args = parser.parse_args()

# check if input arguments are correct
if path.exists(args.videopath):
    cap = cv2.VideoCapture(args.videopath)
else:
    print('404: Video file not found')
    exit()

if path.exists(args.savepath):
    pics_dir = args.savepath
else:
    print('404: Save path not found')
    exit()
frameskip = args.frameskip
if not frameskip == int(frameskip):
    print('ValueError: frameskip value not an int')
    exit()

# Raspicam
#camMatrix = np.array( [[633.06058204 ,  0.0  ,       330.28981083], [  0.0,  631.01252673 ,226.42308878], [  0.0, 0.0,1.        ]])
#distCoefs = np.array([ 5.03468649e-02 ,-4.38421987e-02 ,-2.52895273e-04 , 1.91361583e-03, -4.90955908e-01])

#ananda phone
#camMatrix = np.array( [[630.029356,   0 , 317.89685204], [  0.  ,  631.62683668 ,242.01760626], [  0.  ,  0.,   1.  ]] )
#distCoefs =  np.array([ 0.318628685 ,-2.22790350 ,-0.00156275882 ,-0.00149764901,  4.84589387])
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
bar = progressbar.ProgressBar(max_value=length)
frames = 0
savedframes = 0
lastframe = ''
i = 0
if not args.rewrite:
    filenames = np.sort(np.array(glob.glob(pics_dir + '/*.png')))
    if filenames.shape[0] == 0:
        i = 0
    else:
        i = int(filenames[-1][-9:-4]) + 1

while cap.isOpened() :


    ret, orig_im = cap.read()

    if not ret:
        break
    
    img = cv2.resize(orig_im, (640,480), interpolation = cv2.INTER_CUBIC)
    
    if frames%frameskip == 0:
        filename = pics_dir + '/' + str(i).zfill(5) + '.png'
        cv2.imwrite(filename, img)
        savedframes = savedframes + 1
        lastframe = filename
        i = i + 1
    
    frames = frames + 1
    bar.update(frames)

    if( cv2.waitKey( 1 ) & 0xFF == ord('q') ):
        break;
bar.finish()
print("Number of frames in the video: " + str(length))
print("Number of images saved: " + str(savedframes))
print("Last frame saved: " + lastframe)
        
cap.release()    
#cv2.destroyAllWindows()