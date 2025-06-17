import numpy as np
import argparse
from PIL import Image

def openGrayscaleImage(inputPathname):
  """
    Opens an image file and converts it to a normalized grayscale flattened array.

    Args:
        inputPathname (str): Path to the input image.

    Returns:
        tuple: A tuple containing:
            - Flattened grayscale image as a 1D numpy array normalized between 0 and 1.
            - Width (int) of the image.
            - Height (int) of the image.
  """

  f = np.asarray(Image.open(inputPathname).convert('L'))
  w = np.size(f,1)
  h = np.size(f,0)
  return f.flatten() / 255, w, h

# Parse parameters
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("f0", help="first frame")
parser.add_argument("f1", help="second frame")
parser.add_argument("out", help="output")

args = parser.parse_args()

f1, w, h = openGrayscaleImage(args.f0)
f2, w, h = openGrayscaleImage(args.f1)

diff = f2 - f1
diff = diff-np.min(diff)
diff = diff/np.max(diff)

Image.fromarray(np.uint8(255*np.clip(diff, 0, 1).reshape([h,w])), 'L').save(args.out)