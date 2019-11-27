
"""
Created on Mon Nov 25 2019

@author: Joachim Axelsen, s144015

"""

import cv2
import math
import numpy as np
import time
import argparse
import glob
import os.path
from os import path, mkdir
import progressbar
import random

parser = argparse.ArgumentParser(description='Generate data by pasting clippings onto a set of background images with changing scale.')
parser.add_argument('imagepath', type=str, help='path to background images')
parser.add_argument('clippath', type=str, help='path to the clipped images that are to be pasted onto the background images')
parser.add_argument('savepath', type=str, help='Path of the target folder where the generated data will be saved')
parser.add_argument('--n', type=int, help='integer denoting the number of images to generate for each background image for each clip. Default=5.', default=5)
#parser.add_argument('--rewrite', type=int, help='set to 1 if you want to rewrite the names of the images already in the folder', default=0)

args = parser.parse_args()
minClipRes = 20

# check if input arguments are correct
if path.exists(args.imagepath):
    img_dir = args.imagepath
else:
    exit('404: Image path not found')

if path.exists(args.clippath):
    clip_dir = args.clippath
else:
    exit('404: Clip path not found')
save_dir = args.savepath
if not path.exists(save_dir):
    mkdir(save_dir)
if not args.n == int(args.n):
    exit('ValueError: n value not an int')

# function for overlaying image found on stackoverflow (easier than implementing myself)
# https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
# answered by Mateen Ulhaq
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])



		

bgimgs = np.array(glob.glob(img_dir + '/*.jpg'))
clipimgs = np.array(glob.glob(clip_dir + '/*.png'))
clipimgtxts = np.array(glob.glob(clip_dir + '/*.txt'))



# iterator for file names. 
# TODO check if there are any files in destination folder and use the last name as starting point.
j = 0
bgi = 1
# for each background image
for bgimg in bgimgs:
	origimg = cv2.imread(bgimg)
	#origimg = cv2.resize(origimg, (640,480), interpolation = cv2.INTER_CUBIC)
	origimgh, origimgw, _ = origimg.shape 

	# if image is large enough, instead of scaling, we can cut it into pieces
	# figure out how many pieces it can be cut into:
	nsecw = origimgw/640
	nsech = origimgh/480

	sepw = (math.ceil(nsecw)*640-origimgw)/math.floor(nsecw)
	seph = (math.ceil(nsech)*480-origimgh)/math.floor(nsech)
	cutw = np.array([0])
	cuth = np.array([0],)
	for wi in range(math.ceil(nsecw)-1):
		cutw = np.append(cutw, cutw[-1]+640-sepw)
	for hi in range(math.ceil(nsech)-1):
		cuth = np.append(cuth, cuth[-1]+480-seph)
	print('Processing background image ', bgi, ' out of ', bgimgs.shape[0])
	bgi += 1
	bar = progressbar.ProgressBar(max_value=int(cutw.shape[0]*cuth.shape[0]*clipimgs.shape[0]*args.n))
	for cuty in np.round(cuth).astype(int):
		for cutx in np.round(cutw).astype(int):
			cutimg = origimg[cuty:cuty+480,cutx:cutx+640]
			imgh, imgw, _ = cutimg.shape
			

			# for each clipping image
			#for clipimg,cliptxt in zip(clipimgs, clipimgtxts):
			for i in range(args.n):
				# find random clipping image:
				# OUT COMMENT WHEN MORE CLIPS ARE ADDED!
				randclip = 0 #random.randint(0,clipimgs.shape[0])
				# load the clip image and the clip text descriptor (telling where in the image the drone and the ball is)
				origclip = cv2.imread(clipimgs[randclip], -1)
				origtxt = open(clipimgtxts[randclip], "r")
				lines = origtxt.readlines()

				# find the desired range of scaling based on the size of the clip
				origcliph, origclipw, _ = origclip.shape
				if origcliph < origclipw:
					minScale = minClipRes/origcliph
				elif origclipw < origcliph:
					minScale = minClipRes/origclipw


				img = cutimg.copy()

				# now we need to randomize the clip paste. 
				# First find random scaling. Lower limit is the calculated value and for now the upper limit is the size of the original clip (scale = 1):
				scale = random.uniform(minScale, 1.0)
				scaledH = int(round(scale*origcliph))
				scaledW = int(round(scale*origclipw))

				clip = cv2.resize(origclip, (scaledW,scaledH), interpolation = cv2.INTER_CUBIC)
				cliph, clipw, _ = clip.shape

				# Now we can find a random placement within the borders of the image. For now, the drone has to be completely within the image.
				# the random position found will be where the top left corner of the clip is placed.
				minx = 0
				maxx = imgw-clipw
				miny = 0
				maxy = imgh-cliph
				randx = int(round(random.uniform(minx, maxx)))
				randy = int(round(random.uniform(miny, maxy)))

				overlay_image_alpha(img, 
									clip[:,:,0:3],
									(randx, randy),
									clip[:,:,3]/255.0)

				filename = save_dir + '/' + str(j).zfill(5) + '.txt'
				newF = open(filename, 'w+')
				
				
				for line in lines:

					#print('notice me senpai OwO')

					obj, x, y, w, h = line.split(' ')
					newX = float(x)*clipw+randx
					newY = float(y)*cliph+randy
					newW = float(w)*origclipw*scale
					newH = float(h)*origcliph*scale
					newLine = ' '.join([str(obj),str(newX/imgw),str(newY/imgh),str(newW/imgw),str(newH/imgh)])
					newF.write(newLine + '\n')

					#startpoint = int(newX-(newW/2)),int(newY-(newH/2))
					#endpoint = int(newX+(newW/2)),int(newY+(newH/2))
					
					#img = cv2.rectangle(img, startpoint, endpoint, (255,0,0), 2)

				newF.close()

				#cv2.circle(img, (randx,randy), 1, (0,0,255), 2)

				#cv2.imshow('img', img)
				#cv2.waitKey(0)
				filename = save_dir + '/' + str(j).zfill(5) + '.png'
				cv2.imwrite(filename, img)
				j += 1
				bar.update(bar.value + 1)
				origtxt.close()
	bar.finish()
cv2.destroyWindow('img')



#cv2.imshow('img', img)
#cv2.waitKey(0)

#print(bgimgs)
#print(clipimgs)
