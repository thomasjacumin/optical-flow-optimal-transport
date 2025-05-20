import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

dt = 1#./Nt
dx = 1#./Nx
dy = 1#./Ny

def spaceTimeDiv(u, Nt, Nx, Ny):
  gradTU = np.zeros(Nt*Nx*Ny)
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        if n > 0 and n < Nt-1:
          gradTU[n*Nx*Ny+y*Nx+x] = 0.5*(u[0, (n+1)*Nx*Ny+y*Nx+x] - u[0, (n-1)*Nx*Ny+y*Nx+x])
        elif n == 0:
          gradTU[n*Nx*Ny+y*Nx+x] = u[0, (n+1)*Nx*Ny+y*Nx+x] - u[0, (n)*Nx*Ny+y*Nx+x]
        elif n == Nt-1:
          gradTU[n*Nx*Ny+y*Nx+x] = u[0, (n)*Nx*Ny+y*Nx+x] - u[0, (n-1)*Nx*Ny+y*Nx+x]

  gradXU = np.zeros(Nt*Nx*Ny)
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        if x > 0 and x < Nx-1:
          gradXU[n*Nx*Ny+y*Nx+x] = 0.5*(u[1, n*Nx*Ny+y*Nx+x+1] - u[1, n*Nx*Ny+y*Nx+x-1])
        elif x == 0:
          gradXU[n*Nx*Ny+y*Nx+x] = u[1, n*Nx*Ny+y*Nx+x+1] - u[1, n*Nx*Ny+y*Nx+x]
        elif x == Nx-1:
          gradXU[n*Nx*Ny+y*Nx+x] = u[1, n*Nx*Ny+y*Nx+x] - u[1, n*Nx*Ny+y*Nx+x-1]

  gradYU = np.zeros(Nt*Nx*Ny)
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        if y > 0 and y < Ny-1:
          gradYU[n*Nx*Ny+y*Nx+x] = 0.5*(u[2, n*Nx*Ny+(y+1)*Nx+x] - u[2, n*Nx*Ny+(y-1)*Nx+x])
        elif y == 0:
          gradYU[n*Nx*Ny+y*Nx+x] = u[2, n*Nx*Ny+(y+1)*Nx+x] - u[2, n*Nx*Ny+y*Nx+x]
        elif y == Ny-1:
          gradYU[n*Nx*Ny+y*Nx+x] = u[2, n*Nx*Ny+y*Nx+x] - u[2, n*Nx*Ny+(y-1)*Nx+x]

  return 1./dt*gradTU + 1./dx*gradXU + 1./dy*gradYU

def spaceTimeGrad(u, Nt, Nx, Ny):
  gradTU = np.zeros(Nt*Nx*Ny)
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        if n > 0 and n < Nt-1:
          gradTU[n*Nx*Ny+y*Nx+x] = 0.5*(u[(n+1)*Nx*Ny+y*Nx+x] - u[(n-1)*Nx*Ny+y*Nx+x])
        elif n == 0:
          gradTU[n*Nx*Ny+y*Nx+x] = u[(n+1)*Nx*Ny+y*Nx+x] - u[(n)*Nx*Ny+y*Nx+x]
        elif n == Nt-1:
          gradTU[n*Nx*Ny+y*Nx+x] = u[(n)*Nx*Ny+y*Nx+x] - u[(n-1)*Nx*Ny+y*Nx+x]

  gradXU = np.zeros(Nt*Nx*Ny)
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        if x > 0 and x < Nx-1:
          gradXU[n*Nx*Ny+y*Nx+x] = 0.5*(u[n*Nx*Ny+y*Nx+x+1] - u[n*Nx*Ny+y*Nx+x-1])
        elif x == 0:
          gradXU[n*Nx*Ny+y*Nx+x] = u[n*Nx*Ny+y*Nx+x+1] - u[n*Nx*Ny+y*Nx+x]
        elif x == Nx-1:
          gradXU[n*Nx*Ny+y*Nx+x] = u[n*Nx*Ny+y*Nx+x] - u[n*Nx*Ny+y*Nx+x-1]

  gradYU = np.zeros(Nt*Nx*Ny)
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        if y > 0 and y < Ny-1:
          gradYU[n*Nx*Ny+y*Nx+x] = 0.5*(u[n*Nx*Ny+(y+1)*Nx+x] - u[n*Nx*Ny+(y-1)*Nx+x])
        elif y == 0:
          gradYU[n*Nx*Ny+y*Nx+x] = u[n*Nx*Ny+(y+1)*Nx+x] - u[n*Nx*Ny+y*Nx+x]
        elif y == Ny-1:
          gradYU[n*Nx*Ny+y*Nx+x] = u[n*Nx*Ny+y*Nx+x] - u[n*Nx*Ny+(y-1)*Nx+x]

  return np.array([1./dt*gradTU, 1./dx*gradXU, 1./dy*gradYU])

def stepA(mu, q, rho0, rhoT, r, Nt, Nx, Ny):
  epsilon = 0.001

  spaceTimeDivSM = spaceTimeDiv(mu-r*q, Nt, Nx, Ny)

  F = np.zeros(Nt*Nx*Ny)
  Av = []
  Ax = []
  Ay = []
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        # Not on boundaries (space and time)
        if n>=1 and n<Nt-1 and y>=1 and y<Ny-1 and x>=1 and x<Nx-1:
          # A[n*Nx*Ny+y*Nx+x, (n+1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append((n+1)*Nx*Ny+y*Nx+x)
          Av.append(-r*1./dt**2)
          # A[n*Nx*Ny+y*Nx+x, (n-1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append((n-1)*Nx*Ny+y*Nx+x)
          Av.append(-r*1./dt**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+y*Nx+x+1)
          Av.append(-r*1./dx**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+y*Nx+x-1)
          Av.append(-r*1./dx**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+(y+1)*Nx+x)
          Av.append(-r*1./dy**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+(y-1)*Nx+x)
          Av.append(-r*1./dy**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+y*Nx+x)
          Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon)

          F[n*Nx*Ny+y*Nx+x] = spaceTimeDivSM[n*Nx*Ny+y*Nx+x]
        else:
          # Neumann in time
          if n == 0:
            # A[n*Nx*Ny+y*Nx+x, (n+1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
            Ax.append(n*Nx*Ny+y*Nx+x)
            Ay.append((n+1)*Nx*Ny+y*Nx+x)
            Av.append(-r*1./dt**2)

            F[n*Nx*Ny+y*Nx+x] = spaceTimeDivSM[n*Nx*Ny+y*Nx+x] - 1/dt*(rho0[y*Nx+x]-mu[0,(0)*Nx*Ny+y*Nx+x]+r*q[0,(0)*Nx*Ny+y*Nx+x])

            # Not on space boundaries
            if y>=1 and y<Ny-1 and x>=1 and x<Nx-1:
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x+1)
              Av.append(-r*1./dx**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x-1)
              Av.append(-r*1./dx**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+(y+1)*Nx+x)
              Av.append(-r*1./dy**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+(y-1)*Nx+x)
              Av.append(-r*1./dy**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon +r*1./dt**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x)
              Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon - r*1./dt**2)
            else:
              if x == 0:
                if y == 0: # top-left corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2-r*1./dx**2 +r*1./dt**2 -r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2 -r*1./dx**2-r*1./dy**2) 
                elif y == Ny-1: # bottom-left corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2 +r*1./dt**2-r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2-r*1./dy**2)
                else: # on left but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2+r*1./dt**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dt**2-r*1./dx**2)
              elif x == Nx-1:
                if y == 0: # top-right corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2 -r*1./dy**2+r*1./dt**2-r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dt**2-r*1./dx**2-r*1./dy**2)
                elif y == Ny-1: # bottom-right corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2 -r*1./dy**2+r*1./dt**2-r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2-r*1./dy**2)
                else: # on right but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2+r*1./dt**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2)
              else:
                if y == 0: # on top but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2+r*1./dt**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dt**2-r*1./dy**2)
                elif y == Ny-1: # on bottom but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2+r*1./dt**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2)
          elif n == Nt-1:
            # A[n*Nx*Ny+y*Nx+x, (n-1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
            Ax.append(n*Nx*Ny+y*Nx+x)
            Ay.append((n-1)*Nx*Ny+y*Nx+x)
            Av.append(-r*1./dt**2)

            F[n*Nx*Ny+y*Nx+x] = spaceTimeDivSM[n*Nx*Ny+y*Nx+x ] + 1/dt*(rhoT[y*Nx+x]-mu[0,(Nt-1)*Nx*Ny+y*Nx+x]+r*q[0,(Nt-1)*Nx*Ny+y*Nx+x])
            # Not on space boundaries
            if y>=1 and y<Ny-1 and x>=1 and x<Nx-1:
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x+1)
              Av.append(-r*1./dx**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x-1)
              Av.append(-r*1./dx**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+(y+1)*Nx+x)
              Av.append(-r*1./dy**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+(y-1)*Nx+x)
              Av.append(-r*1./dy**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon +r*1./dt**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x)
              Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2)
            else:
              if x == 0:
                if y == 0: # top-left corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2-r*1./dy**2+r*1./dt**2-r*1./dy**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2-r*1./dx**2)
                elif y == Ny-1: # bottom-left corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2 -r*1./dy**2+r*1./dt**2-r*1./dy**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dt**2-r*1./dy**2-r*1./dx**2)
                else: # on left but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2 +r*1./dt**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2)
              elif x == Nx-1:
                if y == 0: # top-right corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2+r*1./dt**2-r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2-r*1./dy**2)
                elif y == Ny-1: # bottom-right corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2+r*1./dt**2-r*1./dy**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2-r*1./dx**2)
                else: # on right but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2 +r*1./dt**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2)
              else:
                if y == 0: # on top but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 +r*1./dt**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2)
                elif y == Ny-1: # on bottom but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2 +r*1./dt**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2)
          else:
            # not on time boundaries
            F[n*Nx*Ny+y*Nx+x] = spaceTimeDivSM[n*Nx*Ny+y*Nx+x]
            # A[n*Nx*Ny+y*Nx+x, (n+1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
            Ax.append(n*Nx*Ny+y*Nx+x)
            Ay.append((n+1)*Nx*Ny+y*Nx+x)
            Av.append(-r*1./dt**2)
            # A[n*Nx*Ny+y*Nx+x, (n-1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
            Ax.append(n*Nx*Ny+y*Nx+x)
            Ay.append((n-1)*Nx*Ny+y*Nx+x)
            Av.append(-r*1./dt**2)
            if x == 0:
              if y == 0: # top-left corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2-r*1./dx**2-r*1./dx**2-r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2-r*1./dy**2)
              elif y == Ny-1: # bottom-left corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2-r*1./dy**2-r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2-r*1./dx**2)
              else: # on left but not on corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2-r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2)
            elif x == Nx-1:
              if y == 0: # top-right corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2-r*1./dx**2-r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2-r*1./dy**2)
              elif y == Ny-1: # bottom-right corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2-r*1./dy**2-r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2-r*1./dx**2)
              else: # on right but not on corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2-r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2)
            else:
              if y == 0: # on top but not on corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2-r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2)
              elif y == Ny-1: # on bottom but not on corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2-r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2)
  A = sparse.csr_matrix((Av, (Ax, Ay)), shape=[Nt*Nx*Ny, Nt*Nx*Ny])
  return spsolve(A, F) 

def stepB(p, Nt, Nx, Ny):
  a = np.zeros(Nt*Nx*Ny)
  b1 = np.zeros(Nt*Nx*Ny)
  b2 = np.zeros(Nt*Nx*Ny)

  for i in range(Nt*Nx*Ny):
    alpha = p[0,i]
    beta1 = p[1,i]
    beta2 = p[2,i] 

    if 2*alpha + beta1**2 + beta2**2 <= 0:
      a[i] = alpha
      b1[i] = beta1
      b2[i] = beta2
    else:
      # on passe (alpha, beta1, beta2) en coordonnées cylindriques (alpha, rho, theta)
      rho = np.sqrt(beta1**2 + beta2**2)
      theta = np.arctan2(beta2, beta1)
      if -32*(alpha+1)**3-108*rho**2 < 0:
        # print("racine unique")
        zh = -1/3*(alpha + 1)/np.power(1/4*np.sqrt(2)*rho + 1/6*np.sqrt(4/3*alpha**3 + 4*alpha**2 + 9/2*rho**2 + 4*alpha + 4/3), 1/3)
        zh = zh + np.power(1/4*np.sqrt(2)*rho + 1/6*np.sqrt(4/3*alpha**3 + 4*alpha**2 + 9/2*rho**2 + 4*alpha + 4/3), 1/3)
        alphaH = -zh**2
        rhoH = np.sqrt(2)*zh
      else:
        # print("racine triple")
        zh = 2*np.sqrt(2/3)*np.sqrt(-alpha-1)*np.cos(1/3*np.arccos(np.power(3/2, 3/2)*rho/np.power(-alpha-1, 3/2)))
        alphaH = -0.5*zh**2
        rhoH = zh
      # on passe (alphaH, rhoH, theta) en coordonnées carthésiennes (alphaH, beta1H, beta2H)
      beta1H = rhoH*np.cos(theta)
      beta2H = rhoH*np.sin(theta)

      a[i] = alphaH
      b1[i] = beta1H
      b2[i] = beta2H
  return np.array([a, b1, b2])

def solve(rho0, rhoT, Nt, Nx, Ny, r=1, epsilon=0.3, max_it=100):    
    qPrev = np.zeros([3, Nt*Nx*Ny]) # = [a, b] \to\R^3
    mu = np.zeros([3, Nt*Nx*Ny]) # = [rho, m] m\to \R^2
    
    for i in range(0,max_it):
      phi = stepA(mu, qPrev, rho0, rhoT, r, Nt, Nx, Ny)
    
      spaceTimeGradPhi = spaceTimeGrad(phi, Nt, Nx, Ny)
      q = stepB(spaceTimeGradPhi + 1./r*mu, Nt, Nx, Ny)
      # stepC
      muNext = mu + r*( spaceTimeGradPhi - q )
    
      qPrev = q
      mu = muNext
    
      res = spaceTimeGradPhi[0,:] + 0.5*(np.power(spaceTimeGradPhi[1,:], 2) + np.power(spaceTimeGradPhi[2,:], 2) )
      crit = np.sqrt( np.sum( np.multiply(mu[0,:], np.abs(res))) / np.sum( np.multiply(mu[0,:], np.power(spaceTimeGradPhi[1,:], 2) + np.power(spaceTimeGradPhi[2,:], 2) )) )
      print(str(crit)+" ("+str(i+1)+"/"+str(max_it)+")")
    
      if crit <= epsilon:
        break
    return mu, phi, q