import picamera
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
from tqdm import tqdm
from utils import draw_boxes
from frontend import YOLO
import json
import sys
# Keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Reshape, Activation,Input, BatchNormalization, Flatten, Dense, Lambda, Dropout
from keras import optimizers
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image 

with open('config.json') as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

yolo = YOLO(backend             = config['model']['backend'],
        input_size          = config['model']['input_size'], 
        labels              = config['model']['labels'], 
        max_box_per_image   = config['model']['max_box_per_image'],
        anchors             = config['model']['anchors'])

yolo.load_weights('fire_weights.h5')
img_width, img_height = 640, 480
labels = config['model']['labels']


with picamera.PiCamera() as camera:
    camera.resolution = (640, 480) # (320, 240)
    camera.framerate = 24
    frame = np.empty((480, 640, 3), dtype=np.uint8) # (240, 320, 3)
    try:
        while True:
            camera.capture(frame, 'rgb', use_video_port = True)
            frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
            boxes = yolo.predict(frame)
            frame = draw_boxes(frame, boxes, config['model']['labels'])
            cv.imshow('frame', frame)
            cv.waitKey(2)
    except KeyboardInterrupt:
        pass
