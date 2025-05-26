import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse import bmat
from scipy import sparse

import utils

class GLLOpticalFlow(object):
  NAME = "GLL"
  LUMINOSITY = True
  def __init__(self, w=0, h=0):
    self.w = w
    self.h = h
    self.alpha = 0.1

  def setAlpha(self, alpha):
    self.alpha = alpha

  def setLambda(self, lambdap):
    self.lambdap = lambdap
  
  def assemble(self, f1, f2):
    w = self.w
    h = self.h

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

    grad = utils.grad(w,h)
    div = -grad.transpose()
    lap = div@grad

    self.A = bmat([[-alpha*lap+sparse.diags(fx**2), sparse.diags(fx*fy), sparse.diags(-fx*f2)],
                   [sparse.diags(fy*fx), -alpha*lap+sparse.diags(fy**2), sparse.diags(-fy*f2)],
                   [sparse.diags(-f2*fx), sparse.diags(-f2*fy), -lambdap*lap+sparse.diags(f2**2)]]).tocsr()
    
    self.b = np.hstack( (-fx*ft, -fy*ft, f2*ft))
    return self

  def process(self):
    w = self.w
    h = self.h
    x = scipy.sparse.linalg.spsolve(self.A, self.b)
    u = x[0:w*h]
    v = x[w*h:2*w*h]
    m = x[2*w*h:3*w*h]
    return [u, v, m]