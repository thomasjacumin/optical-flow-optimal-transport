import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import logging
import cv2
import csv
import math
import colorsys

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
    return u, v, m