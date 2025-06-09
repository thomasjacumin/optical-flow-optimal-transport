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
from PIL import Image
import math
from scipy import sparse
from scipy.sparse import bmat

import operators

def openGrayscaleImage(inputPathname):
  """
    Opens an image file and converts it to a normalized grayscale flattened array.

    Args:
        inputPathname (str): Path to the input image.

    Returns:
        tuple: A tuple containing:
            - Flattened grayscale image as a 1D numpy array normalized between 0 and 1.
            - Width (int) of the image.
            - Height (int) of the image.
  """

  f = np.asarray(Image.open(inputPathname).convert('L'))
  w = np.size(f,1)
  h = np.size(f,0)
  return f.flatten() / 255, w, h

def reconstructTrajectory(xStart, yStart, u, v, Nx, Ny, Nt):
    """
    Reconstructs the trajectory of a particle given a velocity field over time.

    Args:
        xStart (float): Starting x-coordinate.
        yStart (float): Starting y-coordinate.
        u (np.ndarray): Horizontal velocity field (Nt x Nx*Ny).
        v (np.ndarray): Vertical velocity field (Nt x Nx*Ny).
        Nx (int): Width of the domain.
        Ny (int): Height of the domain.
        Nt (int): Number of time steps.

    Returns:
        list: Displacement in x and y directions as [dx, dy].
    """

    xEnd = xStart
    yEnd = yStart
    for n in range(Nt-1):
        tXEnd = int(xEnd)
        tYEnd = int(yEnd)

        # Clamp to avoid out-of-bounds access
        tXEnd = max(0, min(Nx - 2, tXEnd))
        tYEnd = max(0, min(Ny - 2, tYEnd))

        dX = xEnd - tXEnd
        dY = yEnd - tYEnd

        # Bilinear interpolation weights
        w1 = (1 - dY) * (1 - dX)
        w2 = dX * (1 - dY)
        w3 = dY * dX
        w4 = (1 - dX) * dY

        # Interpolate u and v
        i00 = tYEnd * Nx + tXEnd
        i01 = tYEnd * Nx + tXEnd + 1
        i11 = (tYEnd + 1) * Nx + tXEnd + 1
        i10 = (tYEnd + 1) * Nx + tXEnd

        xEnd += (
            w1 * u[n, i00] +
            w2 * u[n, i01] +
            w3 * u[n, i11] +
            w4 * u[n, i10]
        )
        yEnd += (
            w1 * v[n, i00] +
            w2 * v[n, i01] +
            w3 * v[n, i11] +
            w4 * v[n, i10]
        )

    return [xEnd - xStart, yEnd - yStart]
    # w = Nx
    # h = Ny
    # xEnd = xStart
    # yEnd = yStart
    # for n in range(0, Nt-1):
    #     tXEnd = int(xEnd)
    #     tYEnd = int(yEnd)
    #     dX = xEnd-tXEnd
    #     dY = yEnd-tYEnd
    #     w1 = (1-dY)*(1-dX)
    #     w2 = dX*(1-dY)
    #     w3 = dY*dX
    #     w4 = (1-dX)*dY
        
    #     xEndN = xEnd
    #     yEndN = yEnd
    #     if tXEnd < w-1 and tYEnd < h-1:
    #         xEndN = xEndN + w1*u[n, tYEnd*w + tXEnd]
    #         xEndN = xEndN + w2*u[n, tYEnd*w + tXEnd+1]
    #         xEndN = xEndN + w3*u[n, (tYEnd+1)*w + tXEnd+1]
    #         xEndN = xEndN + w4*u[n, (tYEnd+1)*w + tXEnd]
    #         yEndN = yEndN + w1*v[n, tYEnd*w + tXEnd]
    #         yEndN = yEndN + w2*v[n, tYEnd*w + tXEnd+1]
    #         yEndN = yEndN + w3*v[n, (tYEnd+1)*w + tXEnd+1]
    #         yEndN = yEndN + w4*v[n, (tYEnd+1)*w + tXEnd]
    #     elif tXEnd == w-1 and tYEnd < h-1: # left
    #         xEndN = xEndN + w1*u[n, tYEnd*w + tXEnd]
    #         xEndN = xEndN + w2*u[n, tYEnd*w + tXEnd]
    #         xEndN = xEndN + w3*u[n, (tYEnd+1)*w + tXEnd]
    #         xEndN = xEndN + w4*u[n, (tYEnd+1)*w + tXEnd]
    #         yEndN = yEndN + w1*v[n, tYEnd*w + tXEnd]
    #         yEndN = yEndN + w2*v[n, tYEnd*w + tXEnd]
    #         yEndN = yEndN + w3*v[n, (tYEnd+1)*w + tXEnd]
    #         yEndN = yEndN + w4*v[n, (tYEnd+1)*w + tXEnd]
    #     elif tXEnd < w-1 and tYEnd == h-1: # bottom
    #         xEndN = xEndN + w1*u[n, tYEnd*w + tXEnd]
    #         xEndN = xEndN + w2*u[n, tYEnd*w + tXEnd+1]
    #         xEndN = xEndN + w3*u[n, (tYEnd)*w + tXEnd+1]
    #         xEndN = xEndN + w4*u[n, (tYEnd)*w + tXEnd]
    #         yEndN = yEndN + w1*v[n, tYEnd*w + tXEnd]
    #         yEndN = yEndN + w2*v[n, tYEnd*w + tXEnd+1]
    #         yEndN = yEndN + w3*v[n, (tYEnd)*w + tXEnd+1]
    #         yEndN = yEndN + w4*v[n, (tYEnd)*w + tXEnd]
    
    #     xEnd = xEndN
    #     yEnd = yEndN
    # return [xEnd-xStart, yEnd-yStart]

def opticalflow_from_benamoubrenier(phi, Nt, Nx, Ny, grad, div):
    """
    Computes the optical flow field from the Benamou-Brenier formulation.

    Args:
        phi (np.ndarray): Potential function over time and space.
        Nt (int): Number of time steps.
        Nx (int): Image width.
        Ny (int): Image height.

    Returns:
        tuple: u (horizontal flow), v (vertical flow), m (luminosity).
    """
    
    un = np.zeros([Nt, Nx*Ny])
    vn = np.zeros([Nt, Nx*Ny])
    for n in range(0, Nt-1):
        dU = (grad@phi[n*Nx*Ny:(n+1)*Nx*Ny])
        # [dun, dvn] = spaceGrad(phi, n, Nx, Ny)
        un[n,:] = dU[0:Nx*Ny] # dun
        vn[n,:] = dU[Nx*Ny:2*Nx*Ny] # dvn
    
    u = np.zeros(Nx*Ny)
    v = np.zeros(Nx*Ny)
    for yStart in range(0,Ny):
        for xStart in range(0,Nx):
            [up, vp] = reconstructTrajectory(xStart, yStart, un, vn, Nx, Ny, Nt)
            u[yStart*Nx + xStart] = up
            v[yStart*Nx + xStart] = vp

    print(u.shape)
    print(v.shape)
    print(div.shape)
    m = - div@np.concatenate((u,v)) # spaceDiv(np.array([u,v]), Nx, Ny)
    
    return u, v, m


def apply_opticalflow(f1, u, v, w, h, m=np.array([None])):  
    """
    Applies an optical flow to an image.

    Args:
        f1 (np.ndarray): Flattened input image.
        u (np.ndarray): Horizontal flow vector.
        v (np.ndarray): Vertical flow vector.
        w (int): Image width.
        h (int): Image height.
        m (np.ndarray): Optional luminosity.

    Returns:
        np.ndarray: Transformed image.
    """

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
          tildJ = w-1
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
    """
    Reads a .flo optical flow file.

    Args:
        pathname (str): Path to the .flo file.

    Returns:
        tuple: (width, height, u flow, v flow)
    """

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
    """
    Saves optical flow data to a .flo file.

    Args:
        w (int): Image width.
        h (int): Image height.
        u (np.ndarray): Horizontal flow.
        v (np.ndarray): Vertical flow.
        pathname (str): Output file path.
    """

    f = open(pathname, 'wb')
    np.array([202021.25], dtype=np.float32).tofile(f)
    np.array([w,h], dtype=np.int32).tofile(f)
    data = np.zeros([w*h,2])
    data[:,0] = u
    data[:,1] = v
    # data.reshape([h,w,2])
    np.array(data, dtype=np.float32).tofile(f)

def EE(w, h, u, v, uGT, vGT):
    """
    Calculates the Endpoint Error (EE) and its standard deviation.

    Args:
        w (int): Width.
        h (int): Height.
        u, v (np.ndarray): Computed flow vectors.
        uGT, vGT (np.ndarray): Ground truth flow vectors.

    Returns:
        tuple: (Average EE, Std. Dev. of EE)
    """

    _EE  = np.sqrt( (u-uGT)**2 + (v-vGT)**2 )
    _EE_ignore = []
    for i in range(0, w*h):
        if _EE[i] <= 50:
            _EE_ignore.append(_EE[i])
    AEE  = np.sum( _EE_ignore )/len(_EE_ignore)
    SDEE = np.sqrt( np.sum((_EE_ignore - AEE)**2 )/len(_EE_ignore) )
    return AEE, SDEE

def AE(w, h, u, v, uGT, vGT):
    """
    Computes the Angular Error (AE) and its standard deviation.

    Args:
        w (int): Width.
        h (int): Height.
        u, v (np.ndarray): Computed flow vectors.
        uGT, vGT (np.ndarray): Ground truth flow vectors.

    Returns:
        tuple: (Average AE, Std. Dev. of AE)
    """

    _AE  = np.arccos( (1.0 + u*uGT + v*vGT)/(np.sqrt(1.0 + u**2 + v**2)*np.sqrt(1.0 + uGT**2 + vGT**2)) )
    _AE_ignore = []
    for i in range(0, w*h):
        if math.isnan(_AE[i]) == False:
            _AE_ignore.append(_AE[i])
    AAE  = np.sum( _AE_ignore )/len(_AE_ignore)
    SDAE = np.sqrt( np.sum((_AE_ignore - AAE)**2 )/len(_AE_ignore) )
    return AAE, SDAE

def IE(w, h, I, IGT):
    """
    Computes the Interpolation Error (IE) between the transformed and ground truth images.

    Args:
        w (int): Width.
        h (int): Height.
        I (np.ndarray): Transformed image.
        IGT (np.ndarray): Ground truth image.

    Returns:
        float: Interpolation Error.
    """

    return np.sqrt ( np.sum( (255*I-255*IGT)**2 ) / (w*h) )
