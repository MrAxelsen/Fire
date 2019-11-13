# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:17:15 2019

@author: Joachim Axelsen, s144015, nov-19

TODO: 
grab images from two folders and merge them
remember their labeled txt files
remember train.txt (do valid.txt immediately?)

"""

import cv2
import numpy as np
import time
import argparse
import glob
import os.path
from os import path
import progressbar
from shutil import copyfile

parser = argparse.ArgumentParser(description='merge the data of two paths. creates a new img folder, train.txt file, valid.txt file (optional)')
parser.add_argument('sourcepath1', type=str, help='path to the first source')
parser.add_argument('sourcepath2', type=str, help='path to the second source')
parser.add_argument('destpath', type=str, help='path to the merge destination')
parser.add_argument('--validtxt', type=int, help='set to 1 if you want to create a validation set (default wont create valid.txt)', default=0)

args = parser.parse_args()

# check if input arguments are correct
if path.exists(args.sourcepath1):
    scpath1 = args.sourcepath1
else:
    print('404: Sourcepath1 not found!')
    exit()
if path.exists(args.sourcepath2):
    scpath2 = args.sourcepath2
else:
    print('404: Sourcepath2 not found!')
    exit()
if args.validtxt == 0:
    print('Will not create validation set')
elif args.validtxt == 1:
    print('Creating validation set')
else:
    print('Error: invalid validtxt value: ' + str(args.validtxt))

destpath = args.destpath
if not path.exists(destpath):
    os.mkdir(destpath)
if not path.exists(destpath + '/img'):
    os.mkdir(destpath + '/img')



txt1 = np.array(glob.glob(scpath1 + '/img/*.txt'))
img1 = np.array(glob.glob(scpath1 + '/img/*.png'))
txt2 = np.array(glob.glob(scpath2 + '/img/*.txt'))
img2 = np.array(glob.glob(scpath2 + '/img/*.png'))
n = txt1.shape[0] + txt2.shape[0] + img1.shape[0] + img2.shape[0]
progress = 0
bar = progressbar.ProgressBar(max_value=n)
print('Copying txt files from source 1')

names = [os.path.basename(x) for x in txt1]
for name in names:
    copyfile(scpath1 + name, destpath + name)
    progress = progress + 1
    bar.update(progress)

