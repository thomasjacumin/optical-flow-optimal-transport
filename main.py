import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import utils
import benamou_brenier

f1, w, h = utils.openGrayscaleImage("./data/frame10.png")
f2, w, h = utils.openGrayscaleImage("./data/frame11.png")
Nt = 3
Nx = w
Ny = h
rho0 = f1#0.5*Ny*Nx*f1/np.sum(f1)
rhoT = f2#0.5*Ny*Nx*f2/np.sum(f2)

plt.imshow(rho0.reshape([Ny,Nx]), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(rhoT.reshape([Ny,Nx]), cmap='gray', vmin=0, vmax=1)
plt.show()

mu, phi, q = benamou_brenier.solve(rho0, rhoT, Nt, Nx, Ny)

############################################################################################
xn = mu[0,:].reshape([Nt, Nx*Ny])
for n in range(Nt):
    # for i in range(Nx*Ny):
    #   xn[n, i] = mu[0, n*Nx*Ny + i]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(xn[n,:].reshape([Ny,Nx]), cmap='gray', vmin=0, vmax=1)
    fig.colorbar(im)
    plt.show()
    
    Image.fromarray(np.uint8(255*xn[n,:].reshape([Ny,Nx])), 'L').save("./f_"+str(n)+".png")

############################################################################################
un = np.zeros([Nt, Nx*Ny])
vn = np.zeros([Nt, Nx*Ny])
mn = np.zeros([Nt, Nx*Ny])
for n in range(0, Nt-1):
    [dun, dvn] = utils.spaceGrad(phi, n, Nx, Ny)
    dmn = -utils.spaceDiv(np.array([dun,dvn]), Nx, Ny)
    un[n,:] = dun
    vn[n,:] = dvn
    mn[n,:] = dmn

u = np.zeros(Nx*Ny)
v = np.zeros(Nx*Ny)
m = np.zeros(Nx*Ny)
for yStart in range(0,Ny):
    for xStart in range(0,Nx):
        [up, vp, mp] = utils.reconstructTrajectory(xStart, yStart, un, vn, mn, Nx, Ny, Nt)
        u[yStart*Nx + xStart] = up
        v[yStart*Nx + xStart] = vp
        m[yStart*Nx + xStart] = mp
        
rgb = utils.opticalFlowToRGB(u,v,w,h)
plt.imshow(rgb)
plt.show()

# rec = OFMethod.apply((1+m)*f1, u, v)
# rec = np.clip(rec, 0, 1)
# plt.imshow(rec.reshape([h,w]), vmin=0, vmax=1, cmap='gray')
# plt.show()  

# Image.fromarray(np.uint8(255*f1.reshape([h,w])), 'L').save("./106.png")
# Image.fromarray(np.uint8(255*f2.reshape([h,w])), 'L').save("./107.png")
# Image.fromarray(np.uint8(255*np.clip((m+1)/2, 0, 1).reshape([h,w])), 'L').save("./m_t.png")
# Image.fromarray(np.uint8(255*rec.reshape([h,w])), 'L').save("./x.png")
# Image.fromarray(np.uint8(255*np.abs(rec-f2).reshape([h,w])), 'L').save("./error.png")
# Image.fromarray(np.uint8(255*rgb), 'RGB').save("./u.png")