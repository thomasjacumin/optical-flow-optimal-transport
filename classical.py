# -----------------------------------------------------------------------------
# Copyright (c) 2025, Thomas Jacumin
#
# This file is part of a program licensed under the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import numpy as np
import scipy
import scipy.sparse.linalg
from scipy.sparse import bmat
from scipy import sparse

import utils

class GLLOpticalFlow(object):
  """
    Gennert and Negahdaripour Optical Flow Estimator.
  """
  NAME = "GLL"
  LUMINOSITY = True
  def __init__(self, w=0, h=0):
    """
        Initialize the GLLOpticalFlow object.

        Parameters:
        -----------
        w : int
            Width of the image.
        h : int
            Height of the image.
    """
    self.w = w
    self.h = h
    self.alpha = 0.1

  def setAlpha(self, alpha):
    """
        Set the spatial smoothness parameter.

        Parameters:
        -----------
        alpha : float
            Regularization parameter for smoothness term.
    """
    self.alpha = alpha

  def setLambda(self, lambdap):
    """
        Set the luminosity regularization parameter.

        Parameters:
        -----------
        lambdap : float
            Regularization parameter for luminosity.
    """
    self.lambdap = lambdap
  
  def assemble(self, f1, f2):
    """
        Assemble the sparse linear system Ax = b for optical flow estimation.

        Parameters:
        -----------
        f1 : np.ndarray
            First image frame (grayscale, flattened).
        f2 : np.ndarray
            Second image frame (grayscale, flattened).

        Returns:
        --------
        self : GLLOpticalFlow
            Returns self with system matrix `A` and vector `b` populated.
    """
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
    """
        Solve the assembled system and return optical flow and motion components.

        Returns:
        --------
        [u, v, m] : list of np.ndarray
            u : horizontal component of flow
            v : vertical component of flow
            m : luminosity
    """
    w = self.w
    h = self.h
    x = scipy.sparse.linalg.spsolve(self.A, self.b)
    u = x[0:w*h]
    v = x[w*h:2*w*h]
    m = x[2*w*h:3*w*h]
    return [u, v, m]