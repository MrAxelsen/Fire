from PIL import Image
import cv2
import numpy as np

img = Image.open('objects/droneball/001.png')

(r, g, b, a) = img.split()

print(a.getbbox())