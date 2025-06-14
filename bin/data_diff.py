import numpy as np
import argparse
from PIL import Image

import utils

# Parse parameters
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("f0", help="first frame")
parser.add_argument("f1", help="second frame")
parser.add_argument("out", help="output")

args = parser.parse_args()

f1, w, h = utils.openGrayscaleImage(args.f0)
f2, w, h = utils.openGrayscaleImage(args.f1)

diff = f2 - f1
diff = diff-np.min(diff)
diff = diff/np.max(diff)

Image.fromarray(np.uint8(255*np.clip(diff, 0, 1).reshape([h,w])), 'L').save(args.out)