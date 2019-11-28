# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:17:15 2019

@author: Joachim Axelsen, s144015, nov-15

"""

import numpy as np
import time
import argparse
import glob
import os.path
from os import path

parser = argparse.ArgumentParser(description='Chop video file into frames.')
parser.add_argument('filename', type=str, help='path to file that needs path fixing')
parser.add_argument('pather', type=str, help='path to insert in from of every line')
parser.add_argument('--n', type=int, help='Length of file name of the images. Use this if you need to remove the original path. (default=0 wont remove anything)', default=0)

args = parser.parse_args()

# check if input arguments are correct
if path.exists(args.filename):
    file = args.filename
else:
    exit('404: File not found')
if path.exists(args.pather):
    pics_dir = args.pather
else:
    exit('404: Path not found')
frameskip = args.frameskip
if not frameskip == int(frameskip):
    exit('ValueError: frameskip value not an int')

with open('fixed' + filename, 'r+') as fixed:
    with open(filename, 'r') as reader:
        for line in reader:
            if args.n == 0:
                f.write(pather + line)
            else:
                f.write(pather + line[n:])