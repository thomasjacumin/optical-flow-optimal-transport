import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import utils
import benamou_brenier
import classical

x, w, h = utils.openGrayscaleImage("./results/x.png")
f2, w, h = utils.openGrayscaleImage("./data/Dimetrodon/frame11.png")

r_x = int(w/4)
r_y = int(h/4)
L = 40
for i in range(int(r_y-L/2), int(r_y+L/2)):
    for j in range(int(r_x-L/2), int(r_x+L/2)):
        f2[i+j*w] = f2[i+j*w]-0.2
c_x = int(w/2)
c_y = int(h/2)
R = 30
for i in range(0, h):
    for j in range(0, w):
        if (i-c_x)**2 + (j-c_y)**2 < R**2:
            f2[i+j*w] = f2[i+j*w]+0.2

r_x = int(w/4)
r_y = int(h/4)
L = 60
for i in range(int(r_y-L/2), int(r_y+L/2)):
    for j in range(int(r_x-L/2), int(r_x+L/2)):
        f2[i+j*w] = f2[i+j*w]+0.1
c_x = int(w/2)
c_y = int(2*h/3)
R = 20
for i in range(0, h):
    for j in range(0, w):
        if (i-c_x)**2 + (j-c_y)**2 < R**2:
            f2[i+j*w] = f2[i+j*w]-0.25

print( np.sqrt( np.sum( (x-f2)**2 ) ) )