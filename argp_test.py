import argparse

parser = argparse.ArgumentParser(description='Test the argument parser.')
parser.add_argument('videopath', type=str, help='path to video file that is to be chopped')
parser.add_argument('savepath', type=str, help='path to the folder where to place the chopped frames')
parser.add_argument('--rewrite', type=int, help='set to 1 if you want to rewrite the names of the images already in the folder')

args = parser.parse_args()

vid = args.videopath
folder = args.savepath

print(vid)
print(folder)