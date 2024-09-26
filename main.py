import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import utils
import benamou_brenier

f1, w, h = utils.openGrayscaleImage("./data/frame10-mini.png")
f2, w, h = utils.openGrayscaleImage("./data/frame11-mini.png")

c_x = int(w/2)
c_y = int(h/2)
R = 30
for i in range(0, h):
    for j in range(0, w):
        if (i-c_x)**2 + (j-c_y)**2 < R**2:
            f2[i+j*w] = f2[i+j*w]+0.5

Nt = 4
rho0 = f1#0.5*Ny*Nx*f1/np.sum(f1)
rhoT = f2#0.5*Ny*Nx*f2/np.sum(f2)

plt.imshow(rho0.reshape([h,w]), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(rhoT.reshape([h,w]), cmap='gray', vmin=0, vmax=1)
plt.show()

mu, phi, q = benamou_brenier.solve(rho0, rhoT, Nt, w, h, epsilon=0.1, r=1.1)
u, v, m = utils.opticalflow_from_benamoubrenier(phi, Nt, w, h)

rgb = utils.opticalFlowToRGB(u, v, w, h)
plt.imshow(rgb)
plt.show()

rec = utils.apply_opticalflow(f1, u, v, w, h, m)
rec = np.clip(rec, 0, 1)
# plt.imshow(rec.reshape([h,w]), vmin=0, vmax=1, cmap='gray')
# plt.show()  

Image.fromarray(np.uint8(255*np.clip((m+1)/2, 0, 1).reshape([h,w])), 'L').save("./m_t.png")
Image.fromarray(np.uint8(255*rec.reshape([h,w])), 'L').save("./x.png")
Image.fromarray(np.uint8(255*np.abs(rec-f2).reshape([h,w])), 'L').save("./error.png")
Image.fromarray(np.uint8(255*rgb), 'RGB').save("./results/u.png")