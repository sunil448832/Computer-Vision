from utils import*
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse

parser = argparse.ArgumentParser("give the input")
parser.add_argument('input', help="input image")
parser.add_argument('output_dir', help="output directory to save output image")
parser.add_argument('--foreground', default=None,
                    help="foreground image for composition")
parser.add_argument('--mask', default=-1, help="mask image for composition")
parser.add_argument('--brightness', default=-1, type=int,
                    help='brighten the image with given factor')
parser.add_argument('--contrast', default=-1, type=int,
                    help='contrast the image with given factor')
parser.add_argument('--blur', default=-1, type=int, help='blur the image')
parser.add_argument('--sharpen', default=-1, type=int,
                    help='sharpen the image')
parser.add_argument('--edge_detect', default=-1,
                    type=int, help='detect the edges')
parser.add_argument('--scale', default=-1, type=int, help='image scaling')
parser.add_argument('--composition', default=-1, type=int,
                    help='composition of background and foregraound')
parser.add_argument('--factor', type=float, default=.8,
                    help='factor for luminance')
parser.add_argument('--sigma', type=float, default=.8,
                    help='standerd deviation of gaussaian kernel')
parser.add_argument('--sx', type=float, default=1,
                    help='scaling factor in x direction')
parser.add_argument('--sy', type=float, default=1,
                    help='scaling factor in y direction')
parser.add_argument('--interpolation_method', type=str, default="point",
                    help='interpolation method like point,bilinear,gaussian')
args = parser.parse_args()

arglist = [[args.brightness, 'brightness'], [args.contrast, 'contrast'], [args.blur, 'blur'], [args.sharpen,
                                                                                               'sharpen'], [args.edge_detect, 'edge_detect'], [args.scale, 'scale'], [args.composition, 'composition']]
arglist_ = [[i, j] for i, j in arglist if i > -1]
arg_list = sorted(arglist_, key=lambda x: x[0])


def perform_operation(image, op):

    if op == "brightness":
        return brightness(image, args.factor)

    if op == "contrast":
        return contrast(image, args.factor)

    if op == "blur":
        return blur(image, args.sigma)

    if op == "sharpen":
        return sharpen(image, args.factor)

    if op == "edge_detect":
        return edges(image)

    if op == "scale":
        return scale(image, args.sx, args.sy, args.interpolation_method)

    if op == "composition":
        fore = mpimg.imread(args.foreground).copy()
        mask = mpimg.imread(args.mask).copy()
        return composition(image, fore, mask)


name = ""
image = mpimg.imread(args.input).copy()

for _, op in arg_list:
    if op == 'contrast' or op == "brightness":
        name += op[:3] + str(args.factor) + "_"
    elif op == 'blur':
        name += op[:3] + str(args.sigma) + "_"

    elif op == 'scale':
        name += op[:3] + str(args.sx) + " " + str(args.sy) + \
            " " + args.interpolation_method + "_"

    else:
        name += op[:3] + "_"
    image = perform_operation(image, op)

plt.imshow(image)
plt.savefig(args.output_dir + "/%sout.jpg" % name)
