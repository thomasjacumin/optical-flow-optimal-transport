import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import utils
import benamou_brenier
import classical

f1, w, h = utils.openGrayscaleImage("./data/106.png")
f2, w, h = utils.openGrayscaleImage("./data/107.png")

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

## Benamou-Brenier
mu, phi, q = benamou_brenier.solve(rho0, rhoT, Nt, w, h, epsilon=0.1, r=1, max_it=60)
u, v, m = utils.opticalflow_from_benamoubrenier(phi, Nt, w, h)
xn = mu[0,:].reshape([Nt, w*h])
for n in range(Nt):
  Image.fromarray(np.uint8(255*np.clip(xn[n,:], 0, 1).reshape([h,w])), 'L').save("./results/f_"+str(n)+".png")
rec = utils.apply_opticalflow(f1, u, v, w, h, m)
rec = np.clip(rec, 0, 1)

utils.saveFlo(w, h, u, v,"./results/u-ot.flo")
Image.fromarray(np.uint8(255*np.clip((m+1)/2, 0, 1).reshape([h,w])), 'L').save("./results/m-ot.png")
Image.fromarray(np.uint8(255*rec.reshape([h,w])), 'L').save("./results/x-ot.png")
Image.fromarray(np.uint8(255*np.abs(rec-f2).reshape([h,w])), 'L').save("./results/error-ot.png")

## Classical
classical = classical.GLLOpticalFlow(w,h)
classical.setAlpha(0.1)
classical.setLambda(0.2)
[u, v, m] = classical.assemble(f1, f2).process()
rec = utils.apply_opticalflow(f1, u, v, w, h, m)
rec = np.clip(rec, 0, 1)

utils.saveFlo(w, h, u, v,"./results/u-c.flo")
Image.fromarray(np.uint8(255*np.clip((m+1)/2, 0, 1).reshape([h,w])), 'L').save("./results/m-c.png")
Image.fromarray(np.uint8(255*rec.reshape([h,w])), 'L').save("./results/x-c.png")
Image.fromarray(np.uint8(255*np.abs(rec-f2).reshape([h,w])), 'L').save("./results/error-c.png")