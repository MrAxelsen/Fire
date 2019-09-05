import picamera
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import glob

filenames = np.sort(np.array(glob.glob('images/*.png')))
if filenames.shape[0] == 0:
    i = 0
else:
    i = int(filenames[-1][-9:-4]) + 1

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480) # (320, 240)
    camera.framerate = 24
    frame = np.empty((480, 640, 3), dtype=np.uint8) # (240, 320, 3)
    try:
        while True:
            camera.capture(frame, 'rgb', use_video_port = True)
            frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
            cv.imshow('frame', frame)
            filename = '/home/pi/Fire/images/img' + str(i).zfill(5) + '.png'
            i = i + 1
            cv.imwrite(filename, frame)
            print('Saved image: img' + str(i).zfill(5) + '.png')
            cv.waitKey(2000)
    except KeyboardInterrupt:
        pass

