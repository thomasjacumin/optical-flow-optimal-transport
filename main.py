import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import utils
import benamou_brenier
import classical

f1, w, h = utils.openGrayscaleImage("./data/106.png")
f2, w, h = utils.openGrayscaleImage("./data/107.png")

# r_x = int(w/4)
# r_y = int(h/4)
# L = 40
# for i in range(int(r_y-L/2), int(r_y+L/2)):
#     for j in range(int(r_x-L/2), int(r_x+L/2)):
#         f2[i+j*w] = f2[i+j*w]-0.2
# c_x = int(w/2)
# c_y = int(h/2)
# R = 30
# for i in range(0, h):
#     for j in range(0, w):
#         if (i-c_x)**2 + (j-c_y)**2 < R**2:
#             f2[i+j*w] = f2[i+j*w]+0.2

# r_x = int(w/4)
# r_y = int(h/4)
# L = 60
# for i in range(int(r_y-L/2), int(r_y+L/2)):
#     for j in range(int(r_x-L/2), int(r_x+L/2)):
#         f2[i+j*w] = f2[i+j*w]+0.1
# c_x = int(w/2)
# c_y = int(2*h/3)
# R = 20
# for i in range(0, h):
#     for j in range(0, w):
#         if (i-c_x)**2 + (j-c_y)**2 < R**2:
#             f2[i+j*w] = f2[i+j*w]-0.25


# r_x = int(w/4)
# r_y = int(h/4)
# L = 60
# for i in range(int(r_y-L/2), int(r_y+L/2)):
#     for j in range(int(r_x-L/2), int(r_x+L/2)):
#         f2[i+j*w] = f2[i+j*w]+0.1
# c_x = int(w/2)
# c_y = int(2*h/3)
# R = 40
# for i in range(0, h):
#     for j in range(0, w):
#         if (i-c_x)**2 + (j-c_y)**2 < R**2:
#             f2[i+j*w] = f2[i+j*w]-0.25

c_x = int(3*w/4)
c_y = int(h/4)
R = 30
a = 1
b = 2
for i in range(0, h):
    for j in range(0, w):
        if (i-c_y)**2/a**2 + (j-c_x)**2/b**2 < R**2:
            f2[j+i*w] = f2[j+i*w]+0.2

c_x = int(w/4)
c_y = int(3*h/4)
R = 20
a = 2
b = 1
for i in range(0, h):
    for j in range(0, w):
        if (i-c_y)**2/a**2 + (j-c_x)**2/b**2 < R**2:
            f2[j+i*w] = f2[j+i*w]-0.2


Image.fromarray(np.uint8(255*np.clip(f2, 0, 1).reshape([h,w])), 'L').save("./results/f2.png")
diff = f2-f1
diff = diff - np.min(diff)
diff = diff/np.max(diff)
Image.fromarray(np.uint8(255*np.clip(diff, 0, 1).reshape([h,w])), 'L').save("./results/diff.png")

Nt = 8
rho0 = f1/(np.sum(f1/(w*h)))
rhoT = f2/(np.sum(f2/(w*h)))

plt.imshow(rho0.reshape([h,w]), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(rhoT.reshape([h,w]), cmap='gray', vmin=0, vmax=1)
plt.show()

mu, phi, q = benamou_brenier.solve(rho0, rhoT, Nt, w, h, epsilon=0.1, r=1, max_it=60)
u, v, m = utils.opticalflow_from_benamoubrenier(phi, Nt, w, h)
xn = mu[0,:].reshape([Nt, w*h])
for n in range(Nt):
  Image.fromarray(np.uint8(255*np.clip(xn[n,:], 0, 1).reshape([h,w])), 'L').save("./results/f_"+str(n)+".png")

# classical = classical.GLLOpticalFlow(w,h)
# classical.setAlpha(0.1)
# classical.setLambda(0.2)
# [u, v, m] = classical.assemble(f1, f2).process()

utils.saveFlo(w, h, u, v,"./results/u.flo")

rec = utils.apply_opticalflow(f1, u, v, w, h, m)
rec = np.clip(rec, 0, 1)

Image.fromarray(np.uint8(255*np.clip((m+1)/2, 0, 1).reshape([h,w])), 'L').save("./results/m_t.png")
Image.fromarray(np.uint8(255*rec.reshape([h,w])), 'L').save("./results/x.png")
Image.fromarray(np.uint8(255*np.abs(rec-f2).reshape([h,w])), 'L').save("./results/error.png")