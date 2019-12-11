
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
import sys
import matplotlib.pyplot as plt
import progressbar
import random
from PIL import Image, ImageStat, ImageFont, ImageDraw
from settings import *
sys.path.insert(0, POISSON_BLENDING_DIR)
from pb import *
from pyblur import *

MOTION = 0
objectDict = {'0':'ball', '1':'drone', '2':'balloon'}

minClipRes = 20
class boundingBox:
    def __init__(self, ident, x, y, w, h):
        self.ident = ident
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def __repr__(self):
        return '<bbox x: %s, y: %s>' % (self.x, self.y)
    def __str__(self):
        return 'BoundingBox obj: %s @ x: %s, y: %s' % (self.ident, self.x, self.y)

#boundingbox = namedtuple('boundingbox', 'x y w h')

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


# FROM https://github.com/debidatta/syndata-generation
# modified by me
def overlap(a, b):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed 
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes 
       don't overlap

    Args:
        a(boundingbox): Bounding box 1
        b(boundingbox): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    '''
    dx = min(a.x+a.w, b.x+b.w) - max(a.x, b.x)
    dy = min(a.y+a.h, b.y+b.h) - max(a.y, b.y)
    
    if (dx>=0) and (dy>=0) and float(dx*dy) > MAX_ALLOWED_IOU*(a.x+a.w-a.x)*(a.y+a.h-a.y):
        return True
    else:
        return False

# FROM https://github.com/debidatta/syndata-generation
def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:
        kerneldim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """ 
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])

# FROM https://github.com/debidatta/syndata-generation
def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate 
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes)) 
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img

def parse_args():
    '''Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Generate data by pasting clippings onto a set of background images with changing scale.')
    parser.add_argument('--verbose', type=str, help='Set to 1 to print extra debugging messages. (default=0)', default=0)
    #parser.add_argument('savepath', type=str, help='Path of the target folder where the generated data will be saved')
    parser.add_argument('--n', type=int, help='integer denoting the number of images to generate. Default=5.', default=5)
    parser.add_argument('--noshow', type=int, help='Every second a new image is shown. Set to 1 to stop showing images during generation. Default=show images')
    #parser.add_argument('--rewrite', type=int, help='set to 1 if you want to rewrite the names of the images already in the folder', default=0)

    return parser.parse_args()

# FROM https://github.com/debidatta/syndata-generation
def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

# FROM https://github.com/debidatta/syndata-generation
# modified to include alpha channel
def PIL2array4C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 4)

# FROM https://github.com/debidatta/syndata-generation
def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def cv2pil(image):
    '''
    converts an opencv image to a PIL image
    '''
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def pil2cv(image):
    '''
    converts a PIL image to an opencv image
    '''
    numpy_image=np.array(image) 
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

def hardMask(image):
    '''
    Create a mask where an alpha channel is 255
    '''
    stat = ImageStat.Stat(image)
    im = image.point(lambda p: 255 if p > NONE_THRESHOLD and 255 else 0)
    #im = image.point(lambda p: p > NONE_THRESHOLD and 255)
    stat = ImageStat.Stat(im)
    if stat.sum == 0:
        exit('ERROR: ImageStat.sum equals zero. No values above 190')
    return im

def img2bg():
    '''
    load image
    get random snippet from the image
    resize to (WIDTH, HEIGHT) and return
    since I have always used opencv to resize (and will probably also in the final implementation) the image is loaded and resized with opencv
    '''
    bgimgs = np.array(glob.glob(BACKGROUND_DIR + '*.jpg'))
    randbg = random.randint(0, bgimgs.shape[0]-1)
    # load image
    img = cv2.imread(bgimgs[randbg])
    h, w, _ = img.shape
    # check image size
    if w < WIDTH or h < HEIGHT:
        exit('ERROR: background image: ' + bgpath + ', too small.')
    # find maximum scaling amount (between 1 and ?)
    if w/WIDTH < h/HEIGHT:
        maxscale = w/WIDTH
    else:
        maxscale = h/HEIGHT
    # get random scale within range
    scale = random.uniform(1, maxscale)
    scaledW = round(WIDTH*scale)
    scaledH = round(HEIGHT*scale)
    # get random location within bounds
    x = random.randint(0, w-scaledW)
    y = random.randint(0, h-scaledH)
    return cv2pil(cv2.resize(img[y:y+scaledH, x:x+scaledW, :], (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA))

def drawboxes(bg, bboxes):
	'''
	takes an image (bg) and a list of boundingBoxes and draws the boxes on the image
	'''
    for bbox in bboxes:
        #img = Image.new("RGBA", bg.size, (0,0,0,0))
        draw = ImageDraw.Draw(bg)
        draw.rectangle([(round(bbox.x-(bbox.w/2)), bbox.y-(bbox.h/2)), (bbox.x+(bbox.w/2), bbox.y+(bbox.h/2))], outline=(0,255,0)) # ,fill="black"
        draw.text((round(bbox.x-(bbox.w/2)), bbox.y-(bbox.h/2)), objectDict[bbox.ident], font=ImageFont.truetype("arial.ttf", 10))
    return bg


def blendPaste(image, obj, pos):
    '''
    Pastes an RGBA image onto an RGB image (object onto a background) while applying a random blending algorithm
    Inputs:
    image = RGB background image
    obj = RGBA object image
    pos = offset position that the object is to be pasted on the background image
    Return:
    image = the background image with the object pasted onto
    '''
    global MOTION
    x,y = pos
    randBlend = random.randint(0, len(BLENDING_LIST)-1)
    blend = BLENDING_LIST[randBlend]
    if blend == 'gaussian':
        if args.verbose: print('gaussian')
        image.paste(obj, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(obj.split()[-1]),(5,5),2)))
    elif blend == 'poisson':
        if args.verbose: print('poisson')
        # I NEED POISSON HERE
        offset = (y, x)
        img_mask = PIL2array1C(obj.split()[-1])
        img_src = PIL2array3C(obj.convert('RGB')).astype(np.float64)
        img_target = PIL2array3C(image)
        img_mask, img_src, offset_adj \
             = create_mask(img_mask.astype(np.float64),
                img_target, img_src, offset=offset)
        background_array = poisson_blend(img_mask, img_src, img_target,
                          method='normal', offset_adj=offset_adj)
        image = Image.fromarray(background_array, 'RGB')
    elif blend == 'soft':
        if args.verbose: print('soft')
        image.paste(obj, (x,y), obj)
    elif blend == 'box':
        if args.verbose: print('box')
        image.paste(obj, (x,y), Image.fromarray(cv2.blur(PIL2array1C(obj.split()[-1]),(3,3))))
    elif blend == 'hard':
        if args.verbose: print('hard')
        image.paste(obj, (x,y), hardMask(obj.split()[-1]))
    elif blend == 'motion':
        if args.verbose: print('motion')
        # hvad skal jeg lige gøre her? skal jeg så vente med at motionblur til sidst? måske man ikke skal motionblur ud fra distractors?
        # tror måske bare jeg paster som "soft" og så kan jeg senere motionblur. 
        # hver gang motion blur rammes, adder jeg én til MOTION
        # når jeg har pasted alt kommer det an på hvor mange der har ramt motion
        # hvis to eller mere har ramt motion, så bliver billedet motion blurred
        # hvis kun n rammer motion, så er det 50/50 % chance for motion blur
        image.paste(obj, (x,y), obj)
        MOTION += 1
    return image

def distract(image, disPath):
    global MOTION
    '''
    Pastes a random distractor at a random scale on the given image at a random place
    '''
    w, h = image.size
    # load distractor
    dis = Image.open(disPath)
    disw, dish = dis.size
    # get random scale (first find the limiting dimension)
    if dish < disw:
        minScale = MIN_SCALE/dish
    else:
        minScale = MIN_SCALE/disw
    scale = random.uniform(minScale, 1.0)
    scaledH = int(round(scale*dish))
    scaledW = int(round(scale*disw))
    dis = dis.resize((scaledW,scaledH))
    # get random location
    x = random.randint(0, w-scaledW)
    y = random.randint(0, h-scaledH)
    # paste INSERT RANDOM BLENDING HERE
    image = blendPaste(image, dis, (x, y))
    return image

def Object(image, objPath):
    w, h = image.size
    # load distractor
    obj = Image.open(objPath)
    origw, origh = obj.size
    # get random scale (first find the limiting dimension)
    if origh < origw:
        minScale = MIN_SCALE/origh
    else:
        minScale = MIN_SCALE/origw
    scale = random.uniform(minScale, 1.0)
    scaledH = int(round(scale*origh))
    scaledW = int(round(scale*origw))
    obj = obj.resize((scaledW,scaledH))
    objw, objh = obj.size
    # get random location
    x = random.randint(0, w-scaledW)
    y = random.randint(0, h-scaledH)
    # paste INSERT RANDOM BLENDING HER
    image = blendPaste(image, obj, (x, y))

    # figure out bounding boxes depending on which object
    bboxes = []
    if objPath[-17:-8] == 'droneball':
        if args.verbose: print('generating new bboxes for pasted droneball')
        with open(objPath[:-3]+'txt', 'w+') as f:
            for line in f.readlines():
                ident, oldx, oldy, oldw, oldh = line.split(' ')
                newX = float(oldx)*objw+x
                newY = float(oldy)*objh+y
                newW = float(oldw)*origw*scale
                newH = float(oldh)*origh*scale
                bboxes.append(boundingBox(ident, newX, newY, newW, newH))
    else:
        if args.verbose: print('generating bbox for pasted balloon')
        bboxes.append(boundingBox('2', x+(scaledW/2), y+(scaledH/2), scaledW, scaledH))
    return image, bboxes

def checkOverlap(bboxes, bboxes2):
    '''
    Checks for overlaps between two lists of bounding boxes
    '''
    
    if not bboxes:
        return False

    overlap_detected = False
    for bbox in bboxes:
        for bbox2 in bboxes2:
            if overlap(bbox, bbox2):
                overlap_detected = True
                if args.verbose: print('OVERLAP DETECTED')
    return overlap_detected

def pasteObjects(image):
    '''
    Pastes random objects at random scales on the given image at random places
    '''
    ballimgs = np.array(glob.glob(OBJECTS_DIR + 'droneball/' + '*.png'))
    balloonimgs = np.array(glob.glob(OBJECTS_DIR + 'balloon/' + '*.png'))

    # to keep track of how many (and which) objects to paste an array of random size is created
    # that dictates when to place which objects (so a drone can be behind a balloon and vice versa)
    # this is not very graceful, but it should do the job..
    objOrder = np.array([])
    objNum = random.randint(0,MAX_NO_OF_OBJECTS) # (we want some images with no objects too)
    ballMod = random.randint(1,10)
    if ballMod > BALL_OBJ_CHANCE_MODIFIER and objNum > 0:
        ballInd = random.randint(0,objNum-1)
    else:
        ballInd = 99

    for a in range(objNum):
        if a == ballInd:
            objOrder = np.append(objOrder, ['droneball'])
        else:
            objOrder = np.append(objOrder, ['balloon'])

    bboxes = []
    for i in range(objNum):
        if objOrder[i] == 'droneball': # paste droneball
            if args.verbose: print('pasting a droneball')
            done = False
            a = 0
            while a < MAX_ATTEMPTS_TO_SYNTHESIZE and not done:
                a += 1
                objI = random.randint(0, ballimgs.shape[0]-1)
                image2 = image.copy()
                image2, new_bboxes = Object(image2, ballimgs[objI])
                overlap = checkOverlap(bboxes, new_bboxes)
                if args.verbose: print('overlap: ' + str(overlap))
                if not overlap:
                    image = image2.copy()
                    bboxes.extend(new_bboxes)
                    done = True
        else: # paste balloon
            if args.verbose: print('pasting a balloon')
            done = False
            a = 0
            while a < MAX_ATTEMPTS_TO_SYNTHESIZE and not done:
                a += 1
                objI = random.randint(0, balloonimgs.shape[0]-1)
                image2 = image.copy()
                image2, new_bboxes = Object(image2, balloonimgs[objI])
                if not checkOverlap(bboxes, new_bboxes):
                    image = image2.copy()
                    bboxes.extend(new_bboxes)
                    done = True

    return image, bboxes

def pasteDistractors(image, args):
    distractors = np.array(glob.glob(DISTRACTOR_DIR + '*.png'))
    numDis = random.randint(MIN_NO_OF_DISTRACTOR_OBJECTS, MAX_NO_OF_DISTRACTOR_OBJECTS)
    if args.verbose: print('Generating ' + str(numDis) + ' distractors')
    for d in range(numDis):
        ind = random.randint(0, distractors.shape[0]-1)
        image = distract(image, distractors[ind])
    return image

def create_annofile(image, bboxes, fi):
    imgw, imgh = image.size
    filename = SAVE_DIR + str(fi).zfill(5) + '.txt'
    with open(filename, 'w+') as f:
        for box in bboxes:
            if args.verbose: print(box)
            line = ' '.join([str(box.ident),str(box.x/imgw),str(box.y/imgh),str(box.w/imgw),str(box.h/imgh)])
            f.write(line + '\n')

            

'''
check 0.1. get settings from options-file
check 1. get random background image
check 2. get random cut-out of the background image (scale and placement)
check 3. resize background image to (640, 480)
check 4. get a few random distractors
check 5. paste the distractors at random locations at random size with a random blending method
check 6. create name of the image and create label txt file
check 7. get 0-2 random balloon object images
check 8. paste the balloons at random locations at random sizes and update their bounding boxes in the txt file
    check 8.a. check the overlap of the balloons (must not be more than max overlap)
    check 8.b. if overlap is too big, try again
check 9. get a single random drone/ball object image
check 10. paste the object at random location at random size and update their bounding boxes in the txt file
    check 10.a. check the overlap of the objects (must not be more than max overlap)
    check 10.b. if overlap is too big, try again

TODO:
Sometimes theres a problem with resizing the object paste. Sometimes the numbers are negative?? shouldn't be possible
so there must be an error when determining the smallest dimension before deciding the scaling

'''

def create_synthetic_data(args):
    global MOTION
    
    

    #valid.write(str(j).zfill(5) + '.png\n')
    if not path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    filenames = np.sort(np.array(glob.glob(SAVE_DIR +'*.png')))
    if filenames.shape[0] == 0:
        fi = 1
        traintxt = open('train.txt', 'w+')
    else:
        fi = int(filenames[-1][-9:-4]) + 1
        traintxt = open('train.txt', 'a+')

    waiting = True
    startTime = time.time()
    
    for i in progressbar.progressbar(range(args.n)):
        bg = img2bg()

        #paste random distractors
        bg = pasteDistractors(bg, args)

        bg, bboxes = pasteObjects(bg)

        create_annofile(bg, bboxes, fi)

        # apply motion blur if applicable
        if MOTION >= 2:
            if args.verbose: print('doing linear motion blur')
            bg = LinearMotionBlur3C(PIL2array3C(bg))
        if MOTION == 1:
            ri = random.randint(0,1)
            if ri == 1:
                if args.verbose: print('doing linear motion blur')
                bg = LinearMotionBlur3C(PIL2array3C(bg))
        
        filename = SAVE_DIR + str(fi).zfill(5) + '.png'
        if args.verbose: print('saving as: ' + filename)
        fi += 1
        bg.save(filename)
        traintxt.write(filename + '\n')
        if not args.noshow:
            if waiting:
                if time.time()-startTime > 5:
                    bg = drawboxes(bg, bboxes)
                    img = pil2cv(bg)
                    cv2.imshow('img', img)
                    cv2.waitKey(20)
                    waiting = False
            if not waiting:
                startTime = time.time()
                waiting = True




if __name__ == '__main__':
    args = parse_args()
    create_synthetic_data(args)
