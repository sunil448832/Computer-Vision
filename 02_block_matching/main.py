import cv2
import numpy as np
from matplotlib import pyplot as plt
from util import*
import argparse
import os

parser = argparse.ArgumentParser("give the input")
parser.add_argument('--block_size', nargs='+', help="block size of the patch")
parser.add_argument('--overlapping', type=str, default='yes',
                    help="whether the searching will be overlaped or not")
parser.add_argument('--stride', type=int, default=3,
                    help='number of pixel skipped during searching')
parser.add_argument('--max_serch_lenght', type=int, default=40,
                    help='maximum window size within which to search')
parser.add_argument('--dir_name', default="images",
                    help="make directry to save the output images")
args = parser.parse_args()

# directory of named args.dir_name creation
if not os.path.isdir(args.dir_name):
    os.makedirs(args.dir_name)

vid = cv2.VideoCapture('.\\frame1.gif', 0)
_, frame1 = vid.read()

# plt.imshow(frame1)
# plt.show()

vid2 = cv2.VideoCapture('.\\frame2.gif', 0)
_, frame2 = vid2.read()
# plt.imshow(frame2)
# plt.show()

# to remove the image portion does not comes under the multiple of block size .


def set_image(image, block_size):
    return image[:(image.shape[0]//block_size)*block_size, :(image.shape[1]//block_size)*block_size]


for i in args.block_size:
    if args.overlapping == 'no':
        args.stride = i

    # calculation of the velocity ie displacement vector between frame1 and frame2.
    velocity = velocity_calculation(frame1, frame2, int(
        i), args.stride, args.max_serch_lenght)
    # image reconstruction
    re_image = reconstruction(frame1, velocity, int(i))
    re_image = set_image(re_image, int(i))
    frame2_s = set_image(frame2, int(i))
    # mean square error calculation with respect to frame2
    mse = MSE(re_image, frame2_s)
    plt.imshow(re_image)
    plt.title("reconstruction with block size= %s" % i)
    plt.xlabel("mse error w.r.t frame2 = %.2f " % mse)
    plt.savefig(args.dir_name + "\image_{}".format(i))
    print("reconstruction using of block size  %s  has been calculated ,\
    and mean sqaure error of %.2f has been found" % (i, mse))
