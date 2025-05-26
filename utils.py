import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import logging
import cv2
import csv
import math
import colorsys
from scipy import sparse
from scipy.sparse import bmat

def vectorToColor(w, rhoMax):
  rho = np.sqrt(w[0]**2 + w[1]**2) / rhoMax
  theta = np.arctan2(w[1], w[0])
  if theta < 0:
    theta = theta + 2*math.pi
  return colorsys.hsv_to_rgb(theta / (2*math.pi), np.clip(rho, 0, 1), 1)

def opticalFlowToRGB(u,v,w,h):
  rhoMax = np.sqrt(np.max(np.power(u,2) + np.power(v,2)))
  if rhoMax < 1:
    rhoMax = 1
  oF = np.zeros([w*h,3])
  for i in range(0,w*h):
    color = vectorToColor([u[i],v[i]], rhoMax)
    oF[i,:] = color[:]
  return oF.reshape([h,w,3])

def openGrayscaleImage(inputPathname):
  f = np.asarray(Image.open(inputPathname).convert('L'))
  w = np.size(f,1)
  h = np.size(f,0)
  return f.flatten() / 255, w, h

def spaceDiv(u, Nx, Ny):
  dx = 1
  dy = 1
  gradXU = np.zeros(Nx*Ny)
  for y in range(Ny):
    for x in range(Nx):
      if x > 0 and x < Nx-1:
        gradXU[y*Nx+x] = 0.5*(u[0, y*Nx+x+1] - u[0, y*Nx+x-1])
      elif x == 0:
        gradXU[y*Nx+x] = u[0, y*Nx+x+1] - u[0, y*Nx+x]
      elif x == Nx-1:
        gradXU[y*Nx+x] = u[0, y*Nx+x] - u[0, y*Nx+x-1]
  gradYU = np.zeros(Nx*Ny)
  for y in range(Ny):
    for x in range(Nx):
      if y > 0 and y < Ny-1:
        gradYU[y*Nx+x] = 0.5*(u[1, (y+1)*Nx+x] - u[1, (y-1)*Nx+x])
      elif y == 0:
        gradYU[y*Nx+x] = u[1, (y+1)*Nx+x] - u[1, y*Nx+x]
      elif y == Ny-1:
        gradYU[y*Nx+x] = u[1, y*Nx+x] - u[1, (y-1)*Nx+x]
  return 1./dx*gradXU + 1./dy*gradYU

def spaceGrad(u, n, Nx, Ny):
  dx = 1
  dy = 1
  gradXU = np.zeros(Nx*Ny)
  for y in range(Ny):
    for x in range(Nx):
      if x > 0 and x < Nx-1:
        gradXU[y*Nx+x] = 0.5*(u[n*Nx*Ny+y*Nx+x+1] - u[n*Nx*Ny+y*Nx+x-1])
      elif x == 0:
        gradXU[y*Nx+x] = u[n*Nx*Ny+y*Nx+x+1] - u[n*Nx*Ny+y*Nx+x]
      elif x == Nx-1:
        gradXU[y*Nx+x] = u[n*Nx*Ny+y*Nx+x] - u[n*Nx*Ny+y*Nx+x-1]
  gradYU = np.zeros(Nx*Ny)
  for y in range(Ny):
    for x in range(Nx):
      if y > 0 and y < Ny-1:
        gradYU[y*Nx+x] = 0.5*(u[n*Nx*Ny+(y+1)*Nx+x] - u[n*Nx*Ny+(y-1)*Nx+x])
      elif y == 0:
        gradYU[y*Nx+x] = u[n*Nx*Ny+(y+1)*Nx+x] - u[n*Nx*Ny+y*Nx+x]
      elif y == Ny-1:
        gradYU[y*Nx+x] = u[n*Nx*Ny+y*Nx+x] - u[n*Nx*Ny+(y-1)*Nx+x]
  return np.array([1./dx*gradXU, 1./dy*gradYU])

def reconstructTrajectory(xStart, yStart, u, v, mn, Nx, Ny, Nt):
    w = Nx
    h = Ny
    xEnd = xStart
    yEnd = yStart
    m = 0
    for n in range(0, Nt-1):
        tXEnd = int(xEnd)
        tYEnd = int(yEnd)
        dX = xEnd-tXEnd
        dY = yEnd-tYEnd
        w1 = (1-dY)*(1-dX)
        w2 = dX*(1-dY)
        w3 = dY*dX
        w4 = (1-dX)*dY
        
        xEndN = xEnd
        yEndN = yEnd
        if tXEnd < w-1 and tYEnd < h-1:
            xEndN = xEndN + w1*u[n, tYEnd*w + tXEnd]
            xEndN = xEndN + w2*u[n, tYEnd*w + tXEnd+1]
            xEndN = xEndN + w3*u[n, (tYEnd+1)*w + tXEnd+1]
            xEndN = xEndN + w4*u[n, (tYEnd+1)*w + tXEnd]
            yEndN = yEndN + w1*v[n, tYEnd*w + tXEnd]
            yEndN = yEndN + w2*v[n, tYEnd*w + tXEnd+1]
            yEndN = yEndN + w3*v[n, (tYEnd+1)*w + tXEnd+1]
            yEndN = yEndN + w4*v[n, (tYEnd+1)*w + tXEnd]
            m = m + w1*mn[n, tYEnd*w + tXEnd]
            m = m + w2*mn[n, tYEnd*w + tXEnd+1]
            m = m + w3*mn[n, (tYEnd+1)*w + tXEnd+1]
            m = m + w4*mn[n, (tYEnd+1)*w + tXEnd]
        elif tXEnd == w-1 and tYEnd < h-1: # left
            xEndN = xEndN + w1*u[n, tYEnd*w + tXEnd]
            xEndN = xEndN + w2*u[n, tYEnd*w + tXEnd]
            xEndN = xEndN + w3*u[n, (tYEnd+1)*w + tXEnd]
            xEndN = xEndN + w4*u[n, (tYEnd+1)*w + tXEnd]
            yEndN = yEndN + w1*v[n, tYEnd*w + tXEnd]
            yEndN = yEndN + w2*v[n, tYEnd*w + tXEnd]
            yEndN = yEndN + w3*v[n, (tYEnd+1)*w + tXEnd]
            yEndN = yEndN + w4*v[n, (tYEnd+1)*w + tXEnd]
            m = m + w1*mn[n, tYEnd*w + tXEnd]
            m = m + w2*mn[n, tYEnd*w + tXEnd]
            m = m + w3*mn[n, (tYEnd+1)*w + tXEnd]
            m = m + w4*mn[n, (tYEnd+1)*w + tXEnd]
        elif tXEnd < w-1 and tYEnd == h-1: # bottom
            xEndN = xEndN + w1*u[n, tYEnd*w + tXEnd]
            xEndN = xEndN + w2*u[n, tYEnd*w + tXEnd+1]
            xEndN = xEndN + w3*u[n, (tYEnd)*w + tXEnd+1]
            xEndN = xEndN + w4*u[n, (tYEnd)*w + tXEnd]
            yEndN = yEndN + w1*v[n, tYEnd*w + tXEnd]
            yEndN = yEndN + w2*v[n, tYEnd*w + tXEnd+1]
            yEndN = yEndN + w3*v[n, (tYEnd)*w + tXEnd+1]
            yEndN = yEndN + w4*v[n, (tYEnd)*w + tXEnd]
            m = m + w1*mn[n, tYEnd*w + tXEnd]
            m = m + w2*mn[n, tYEnd*w + tXEnd+1]
            m = m + w3*mn[n, (tYEnd)*w + tXEnd+1]
            m = m + w4*mn[n, (tYEnd)*w + tXEnd]
    
        xEnd = xEndN
        yEnd = yEndN
    return [xEnd-xStart, yEnd-yStart, m]

def opticalflow_from_benamoubrenier(phi, Nt, Nx, Ny):
    un = np.zeros([Nt, Nx*Ny])
    vn = np.zeros([Nt, Nx*Ny])
    mn = np.zeros([Nt, Nx*Ny])
    for n in range(0, Nt-1):
        [dun, dvn] = spaceGrad(phi, n, Nx, Ny)
        dmn = -spaceDiv(np.array([dun,dvn]), Nx, Ny)
        un[n,:] = dun
        vn[n,:] = dvn
        mn[n,:] = dmn
    
    u = np.zeros(Nx*Ny)
    v = np.zeros(Nx*Ny)
    m = np.zeros(Nx*Ny)
    for yStart in range(0,Ny):
        for xStart in range(0,Nx):
            [up, vp, mp] = reconstructTrajectory(xStart, yStart, un, vn, mn, Nx, Ny, Nt)
            u[yStart*Nx + xStart] = up
            v[yStart*Nx + xStart] = vp
            m[yStart*Nx + xStart] = mp
    m = -spaceDiv(np.array([u,v]), Nx, Ny)
    return u, v, m


def apply_opticalflow(f1, u, v, w, h, m=np.array([None])):  
    if m.all() != None:
      f1 = (1+m)*f1
    
    x = np.zeros([w*h])
    for i in range(0,h):
      for j in range(0,w):
        tildI = i - v[i*w+j]
        tildJ = j - u[i*w+j]
        dI = tildI-int(tildI)
        dJ = tildJ-int(tildJ)

        w1 = (1-dI)*(1-dJ)
        w2 = dJ*(1-dI)
        w3 = dI*dJ
        w4 = (1-dJ)*dI

        if tildI >= h:
          tildI = h-1
        if tildJ >= w:
          tileJ = w-1
        if tildI < 0:
          tildI = 0
        if tildJ < 0:
          tildJ = 0

        if int(tildI)*w+int(tildJ)<w*h and i*w+j<w*h:
          if int(tildI) < h-1 and int(tildJ) < w-1: # not on the left or bottom boundaries
            x[i*w+j] = w1*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w2*f1[int(tildI)*w + int(tildJ)+1]
            x[i*w+j] = x[i*w+j] + w3*f1[int(tildI+1)*w + int(tildJ)+1]
            x[i*w+j] = x[i*w+j] + w4*f1[int(tildI+1)*w + int(tildJ)]
          elif int(tildI) < h-1 and int(tildJ) == w-1: # left boundary
            x[i*w+j] = w1*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w2*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w3*f1[int(tildI+1)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w4*f1[int(tildI+1)*w + int(tildJ)]
          elif int(tildI) == h-1 and int(tildJ) < w-1: # bottom boundary
            x[i*w+j] = w1*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w2*f1[int(tildI)*w + int(tildJ)+1]
            x[i*w+j] = x[i*w+j] + w3*f1[int(tildI)*w + int(tildJ)+1]
            x[i*w+j] = x[i*w+j] + w4*f1[int(tildI)*w + int(tildJ)]
          else: # bottom-left corner
            x[i*w+j] = w1*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w2*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w3*f1[int(tildI)*w + int(tildJ)]
            x[i*w+j] = x[i*w+j] + w4*f1[int(tildI)*w + int(tildJ)]
    return x

def openFlo(pathname):
    f = open(pathname, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32)
    data_2D = np.reshape(data, newshape=(h,w,2));
    x = data_2D[...,0].flatten()
    y = data_2D[...,1].flatten()
    return w, h, x, y

def saveFlo(w,h,u,v,pathname):
    f = open(pathname, 'wb')
    np.array([202021.25], dtype=np.float32).tofile(f)
    np.array([w,h], dtype=np.int32).tofile(f)
    data = np.zeros([w*h,2])
    data[:,0] = u
    data[:,1] = v
    # data.reshape([h,w,2])
    np.array(data, dtype=np.float32).tofile(f)

def EE(w, h, u, v, uGT, vGT):
    _EE  = np.sqrt( (u-uGT)**2 + (v-vGT)**2 )
    _EE_ignore = []
    for i in range(0, w*h):
        if _EE[i] <= 50:
            _EE_ignore.append(_EE[i])
    AEE  = np.sum( _EE_ignore )/len(_EE_ignore)
    SDEE = np.sqrt( np.sum((_EE_ignore - AEE)**2 )/len(_EE_ignore) )
    return AEE, SDEE

def AE(w, h, u, v, uGT, vGT):
    _AE  = np.arccos( (1.0 + u*uGT + v*vGT)/(np.sqrt(1.0 + u**2 + v**2)*np.sqrt(1.0 + uGT**2 + vGT**2)) )
    _AE_ignore = []
    for i in range(0, w*h):
        if math.isnan(_AE[i]) == False:
            _AE_ignore.append(_AE[i])
    AAE  = np.sum( _AE_ignore )/len(_AE_ignore)
    SDAE = np.sqrt( np.sum((_AE_ignore - AAE)**2 )/len(_AE_ignore) )
    return AAE, SDAE

def IE(w, h, I, IGT):
    return np.sqrt ( np.sum( (255*I-255*IGT)**2 ) / (w*h) )

def grad_x(w,h):
  Av = []
  Ax = []
  Ay = []

  for i in range(0,h):
    for j in range(0,w):
      n = i*w+j
      if j <= w-2:
        Ax.append(n)
        Ay.append(n)
        Av.append(-1)
        #
        Ax.append(n)
        Ay.append(n+1)
        Av.append(1)

  return sparse.csr_matrix((Av, (Ax, Ay)), shape=[w*h, w*h])

def grad_y(w,h):
  Av = []
  Ax = []
  Ay = []

  for i in range(0,h):
    for j in range(0,w):
      n = i*w+j
      if i <= h-2:
        Ax.append(n)
        Ay.append(n)
        Av.append(-1)
        #
        Ax.append(n)
        Ay.append(n+w)
        Av.append(1)

  return sparse.csr_matrix((Av, (Ax, Ay)), shape=[w*h, w*h])

def grad(w,h):
  x = grad_x(w,h)
  y = grad_y(w,h)
  return  bmat([ [x], [y] ])