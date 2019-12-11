# Paths
# Fill this according to own setup
BACKGROUND_DIR = 'backgrounds/'
DISTRACTOR_DIR = 'distractors/'
OBJECTS_DIR = 'objects/'
SAVE_DIR = 'genData/'

POISSON_BLENDING_DIR = '/home/aut/joax/github/pb'

# generation parameters
BLENDING_LIST = ['gaussian','poisson', 'soft', 'box', 'hard', 'motion']
NONE_THRESHOLD = 190
MAX_NO_OF_OBJECTS = 4
BALL_OBJ_CHANCE_MODIFIER = 3 # 3 = about 30% of images have no ball
MIN_NO_OF_DISTRACTOR_OBJECTS = 1
MAX_NO_OF_DISTRACTOR_OBJECTS = 4
WIDTH = 640
HEIGHT = 480
MAX_ATTEMPTS_TO_SYNTHESIZE = 10 # when to give up on correcting the overlaid objects

# Parameters for objects in images
MIN_SCALE = 0.25 # min scale for scale augmentation
MAX_SCALE = 1.0 # max scale for scale augmentation
MAX_ALLOWED_IOU = 0.75 # IOU > MAX_ALLOWED_IOU is considered an occlusion
MIN_WIDTH = 6 # Minimum width of object to use for data generation
MIN_HEIGHT = 6 # Minimum height of object to use for data generation
