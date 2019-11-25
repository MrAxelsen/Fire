
"""
Created on Mon Nov 25 2019

@author: Joachim Axelsen, s144015

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

