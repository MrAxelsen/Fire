import picamera
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import glob

filenames = np.sort(np.array(glob.glob('images/*.png')))

for filename in filenames:
	img = cv.imread(filename)
	imgnumber = filename[-9:-4]
	cv.imwrite(filename, cv.cvtColor(img,cv.COLOR_RGB2BGR))
	print("Converting img: " + str(imgnumber))
