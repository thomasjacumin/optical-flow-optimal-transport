import numpy as np
import scipy
import scipy.sparse.linalg

class GLLOpticalFlow(object):
  NAME = "GLL"
  LUMINOSITY = True
  def __init__(self, w=0, h=0):
    self.w = w
    self.h = h
    self.rho = 0
    self.alpha = 0.1

  def setRho(self, rho):
    self.rho = rho

  def setAlpha(self, alpha):
    self.alpha = alpha

  def setLambda(self, lambdap):
    self.lambdap = lambdap
  
  def assemble(self, f1, f2):
    w = self.w
    h = self.h

    rho = self.rho
    alpha = self.alpha
    lambdap = self.lambdap

    fx = np.zeros(w*h)
    for i in range(0,h):
      for j in range(1,w-1):
        fx[i*w+j] = 0.5*(f2[i*w+j+1]-f2[i*w+j-1])

    fy = np.zeros(w*h)
    for i in range(1,h-1):
      for j in range(0,w):
        fy[i*w+j] = 0.5*(f2[(i+1)*w+j]-f2[(i-1)*w+j])

    ft = f2 - f1

    a1 = fx**2
    a2 = fx*fy
    a3 = -fx*f2
    a4 = fy*fx
    a5 = fy**2
    a6 = -fy*f2
    a7 = -f2*fx
    a8 = -f2*fy
    a9 = f2**2
    
    b1 = -fx*ft
    b2 = -fy*ft
    b3 = f2*ft

    Av = []
    Ax = []
    Ay = []
    b = np.zeros(3*w*h)
    for i in range(0,h):
      for j in range(0,w):
        b[i*w + j]         = b1[i*w+j]
        b[w*h + i*w + j]   = b2[i*w+j]
        b[2*w*h + i*w + j] = b3[i*w+j]

        if i >= 1 and i <= h-2 and j >= 1 and j <= w-2: # not on the boundaries
          # For u
          ## u_i,j
          Av.append(a1[i*w + j] + 4*alpha)
          Ax.append(i*w + j)
          Ay.append(i*w + j)
          ## u_i+1,j
          Av.append(-alpha)
          Ax.append(i*w + j)
          Ay.append((i+1)*w + j)
          ## u_i-1,j
          Av.append(-alpha)
          Ax.append(i*w + j)
          Ay.append((i-1)*w + j)
          ## u_i,j+1
          Av.append(-alpha)
          Ax.append(i*w + j)
          Ay.append(i*w + j+1)
          ## u_i,j-1
          Av.append(-alpha)
          Ax.append(i*w + j)
          Ay.append(i*w + j-1)
          ## v_i,j
          Av.append(a2[i*w + j])
          Ax.append(i*w + j)
          Ay.append(w*h + i*w + j)
          ## m_i,j
          Av.append(a3[i*w + j])
          Ax.append(i*w + j)
          Ay.append(2*w*h + i*w + j)

          # For v
          ## v_i,j
          Av.append(a5[i*w + j] + 4*alpha)
          Ax.append(w*h + i*w + j)
          Ay.append(w*h + i*w + j)
          ## v_i+1,j
          Av.append(-alpha)
          Ax.append(w*h + i*w + j)
          Ay.append(w*h + (i+1)*w + j)
          ## v_i-1,j
          Av.append(-alpha)
          Ax.append(w*h + i*w + j)
          Ay.append(w*h + (i-1)*w + j)
          ## v_i,j+1
          Av.append(-alpha)
          Ax.append(w*h + i*w + j)
          Ay.append(w*h + i*w + j+1)
          ## v_i,j-1
          Av.append(-alpha)
          Ax.append(w*h + i*w + j)
          Ay.append(w*h + i*w + j-1)
          ## u_i,j
          Av.append(a4[i*w + j])
          Ax.append(w*h + i*w + j)
          Ay.append(i*w + j)
          ## m_i,j
          Av.append(a6[i*w + j])
          Ax.append(i*w + j)
          Ay.append(2*w*h + i*w + j)

          # For m
          ## m_i,j
          Av.append(a9[i*w + j] + 4*lambdap)
          Ax.append(2*w*h + i*w + j)
          Ay.append(2*w*h + i*w + j)
          ## m_i+1,j
          Av.append(-lambdap)
          Ax.append(2*w*h + i*w + j)
          Ay.append(2*w*h + (i+1)*w + j)
          ## m_i-1,j
          Av.append(-lambdap)
          Ax.append(2*w*h + i*w + j)
          Ay.append(2*w*h + (i-1)*w + j)
          ## m_i,j+1
          Av.append(-lambdap)
          Ax.append(2*w*h + i*w + j)
          Ay.append(2*w*h + i*w + j+1)
          ## m_i,j-1
          Av.append(-lambdap)
          Ax.append(2*w*h + i*w + j)
          Ay.append(2*w*h + i*w + j-1)
          ## u_i,j
          Av.append(a7[i*w + j])
          Ax.append(2*w*h + i*w + j)
          Ay.append(i*w + j)
          ## v_i,j
          Av.append(a8[i*w + j])
          Ax.append(2*w*h + i*w + j)
          Ay.append(w*h + i*w + j)
        else: # Neumann BC
          if (i==0 and j==0) or (i==0 and j==w-1) or (i==h-1 and j==0) or (i==h-1 and j==w-1): # on a corner
            # For u
            #A[i*w + j, i*w + j] = -2
            Av.append(a1[i*w + j] + 2*alpha)
            Ax.append(i*w + j)
            Ay.append(i*w + j)
            ## v_i,j
            Av.append(a2[i*w + j])
            Ax.append(i*w + j)
            Ay.append(w*h + i*w + j)
            ## m_i,j
            Av.append(a3[i*w + j])
            Ax.append(i*w + j)
            Ay.append(2*w*h + i*w + j)
            if (i==0 and j == 0):
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i+1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j+1)
            elif (i==0 and j == w-1):
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i+1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j-1)
            elif (i==h-1 and j == 0):
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j+1)
            else:
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i-1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j-1)

            # For v
            #A[i*w + j, i*w + j] = -2
            Av.append(a5[i*w + j] + 2*alpha)
            Ax.append(w*h + i*w + j)
            Ay.append(w*h + i*w + j)
            ## u_i,j
            Av.append(a4[i*w + j])
            Ax.append(w*h + i*w + j)
            Ay.append(i*w + j)
            ## m_i,j
            Av.append(a6[i*w + j])
            Ax.append(w*h + i*w + j)
            Ay.append(2*w*h + i*w + j)
            if (i==0 and j == 0):
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h + i*w + j)
              Ay.append(w*h + (i+1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(w*h + i*w + j)
              Ay.append(w*h + i*w + j+1)
            elif (i==0 and j == w-1):
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h + i*w + j)
              Ay.append(w*h + (i+1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(w*h + i*w + j)
              Ay.append(w*h + i*w + j-1)
            elif (i==h-1 and j == 0):
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h + i*w + j)
              Ay.append(w*h + (i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(w*h + i*w + j)
              Ay.append(w*h + i*w + j+1)
            else:
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h + i*w + j)
              Ay.append(w*h + (i-1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(w*h + i*w + j)
              Ay.append(w*h + i*w + j-1)

            # For m
            #A[i*w + j, i*w + j] = -2
            Av.append(a9[i*w + j] + 2*lambdap)
            Ax.append(2*w*h + i*w + j)
            Ay.append(2*w*h + i*w + j)
            ## u_i,j
            Av.append(a7[i*w + j])
            Ax.append(2*w*h + i*w + j)
            Ay.append(i*w + j)
            ## v_i,j
            Av.append(a8[i*w + j])
            Ax.append(2*w*h + i*w + j)
            Ay.append(w*h + i*w + j)
            if (i==0 and j == 0):
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h + i*w + j)
              Ay.append(2*w*h + (i+1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h + i*w + j)
              Ay.append(2*w*h + i*w + j+1)
            elif (i==0 and j == w-1):
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h + i*w + j)
              Ay.append(2*w*h + (i+1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h + i*w + j)
              Ay.append(2*w*h + i*w + j-1)
            elif (i==h-1 and j == 0):
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h + i*w + j)
              Ay.append(2*w*h + (i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h + i*w + j)
              Ay.append(2*w*h + i*w + j+1)
            else:
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h + i*w + j)
              Ay.append(2*w*h + (i-1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h + i*w + j)
              Ay.append(2*w*h + i*w + j-1)
          else: # not on a corner
            # For u
            #A[i*w + j, i*w + j] = -3
            Av.append(a1[i*w + j] + 3*alpha)
            Ax.append(i*w + j)
            Ay.append(i*w + j)
            ## v_i,j
            Av.append(a2[i*w + j])
            Ax.append(i*w + j)
            Ay.append(w*h + i*w + j)
            ## m_i,j
            Av.append(a3[i*w + j])
            Ax.append(i*w + j)
            Ay.append(2*w*h + i*w + j)
            if i == 0: # top
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i+1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j+1)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j-1)
            if i == h-1: # bottom
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j+1)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j-1)
            if j == 0: # left
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i+1)*w + j)
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j+1)
            if j == w-1: # right
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i+1)*w + j)
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append((i-1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(i*w + j)
              Ay.append(i*w + j-1)

            # For v
            #A[i*w + j, i*w + j] = -3
            Av.append(a5[i*w + j] + 3*alpha)
            Ax.append(w*h +i*w + j)
            Ay.append(w*h +i*w + j)
            ## u_i,j
            Av.append(a4[i*w + j])
            Ax.append(w*h +i*w + j)
            Ay.append(i*w + j)
            ## m_i,j
            Av.append(a6[i*w + j])
            Ax.append(w*h +i*w + j)
            Ay.append(2*w*h +i*w + j)
            if i == 0: # top
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +(i+1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +i*w + j+1)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +i*w + j-1)
            if i == h-1: # bottom
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +(i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +i*w + j+1)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +i*w + j-1)
            if j == 0: # left
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +(i+1)*w + j)
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +(i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +i*w + j+1)
            if j == w-1: # right
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +(i+1)*w + j)
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +(i-1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-alpha)
              Ax.append(w*h +i*w + j)
              Ay.append(w*h +i*w + j-1)

            # For m
            #A[i*w + j, i*w + j] = -3
            Av.append(a9[i*w + j] + 3*lambdap)
            Ax.append(2*w*h +i*w + j)
            Ay.append(2*w*h +i*w + j)
            ## u_i,j
            Av.append(a7[i*w + j])
            Ax.append(2*w*h +i*w + j)
            Ay.append(i*w + j)
            ## m_i,j
            Av.append(a8[i*w + j])
            Ax.append(2*w*h +i*w + j)
            Ay.append(w*h +i*w + j)
            if i == 0: # top
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +(i+1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +i*w + j+1)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +i*w + j-1)
            if i == h-1: # bottom
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +(i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +i*w + j+1)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +i*w + j-1)
            if j == 0: # left
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +(i+1)*w + j)
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +(i-1)*w + j)
              #A[i*w + j, i*w + j+1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +i*w + j+1)
            if j == w-1: # right
              #A[i*w + j, (i+1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +(i+1)*w + j)
              #A[i*w + j, (i-1)*w + j] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +(i-1)*w + j)
              #A[i*w + j, i*w + j-1] = 1
              Av.append(-lambdap)
              Ax.append(2*w*h +i*w + j)
              Ay.append(2*w*h +i*w + j-1)

    Av = np.array(Av)
    Ax = np.array(Ax)
    Ay = np.array(Ay)

    self.A = scipy.sparse.csr_matrix( (Av, (Ax, Ay)), shape=(3*w*h, 3*w*h))
    self.b = np.array(b)
    return self

  def process(self):
    w = self.w
    h = self.h
    x = scipy.sparse.linalg.spsolve(self.A, self.b)
    u = x[0:w*h]
    v = x[w*h:2*w*h]
    m = x[2*w*h:3*w*h]
    return [u, v, m]