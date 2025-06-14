import numpy as np
import argparse
from PIL import Image
import random

import utils

def add_rectangle(f, L_x, L_y, r_x, r_y, v):
    res = f
    for j in range(int(r_y-L_y/2), int(r_y+L_y/2)):
        for i in range(int(r_x-L_x/2), int(r_x+L_x/2)):
            res[i+j*w] += v
    return res

def add_circle(f, R, c_x, c_y, v):
    res = f
    for j in range(0, h):
        for i in range(0, w):
            if (i-c_x)**2 + (j-c_y)**2 < R**2:
                res[i+j*w] += v
    return res

def add_random_rectangle(f, w, h):
    L_x = random.randint(10, w-1)
    L_y = random.randint(10, h-1)
    r_x = random.randint(int(L_x/2), int(w-L_x/2))
    r_y = random.randint(int(L_y/2), int(h-L_y/2))
    v = random.uniform(-0.25, 0.25)

    return add_rectangle(f, L_x, L_y, r_x, r_y, v)

def add_random_circle(f, w, h):
    R = random.randint(10, min(w, h))/2
    c_x = random.randint(int(R), int(w-R))
    c_y = random.randint(int(R), int(h-R))
    
    v = random.uniform(-0.25, 0.25)
    return add_circle(f, R, c_x, c_y, v)

# Parse parameters
parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("f", help="frame")
parser.add_argument("out", help="output")
parser.add_argument("seed", type=int, help="random seed")

args = parser.parse_args()

random.seed(args.seed)

f, w, h = utils.openGrayscaleImage(args.f)

f_lum = f
f_lum = add_random_rectangle(f_lum, w, h)
f_lum = add_random_rectangle(f_lum, w, h)
f_lum = add_random_circle(f_lum, w, h)
f_lum = add_random_circle(f_lum, w, h)

Image.fromarray(np.uint8(255*np.clip(f_lum, 0, 1).reshape([h,w])), 'L').save(args.out)