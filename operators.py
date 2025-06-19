from scipy import sparse
from scipy.sparse import bmat
import numpy as np

###################################

def grad_1d_forward_weird(n, h, bc):
    if not bc in ['N']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-np.ones(n), np.ones(n)]
    offsets = [0, 1]
    L = sparse.diags(diagonals, offsets, shape=(n, n+1), format='lil')
    L /= h    
    return L.tocsr()

def grad_1d_backward_weird(n, h, bc):
    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-np.ones(n-1), np.ones(n)]
    offsets = [-1, 0]
    L = sparse.diags(diagonals, offsets, shape=(n, n), format='lil')
    L /= h
    
    L[0, 1]   = 1
    L[-1, -2] = -1
    
    return L.tocsr()

def grad_1d_central_reflexion(n, h, bc):
    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-0.5*np.ones(n-1), 0.5*np.ones(n-1)]
    offsets = [-1, 1]
    L = sparse.diags(diagonals, offsets, shape=(n, n), format='lil')
    L /= h

    L[0, 1]   = 1
    L[-1, -2] = -1
    
    return L.tocsr()

##################################### FD Schemes #####################################

def grad_1d_central(n, h, bc):
    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-0.5*np.ones(n-1), 0.5*np.ones(n-1)]
    offsets = [-1, 1]
    L = sparse.diags(diagonals, offsets, shape=(n, n), format='lil')
    L /= h

    if bc == 'N':
        L[0, 1]   = 0
        L[-1, -2] = 0
    if bc == 'D': # v_0 = v_N = 0 and v_0 = (v_-1 + v_1) / 2
        L[0, 1]   = 1
        L[-1, -2] = -1

    return L.tocsr()

def grad_1d_forward(n, h, bc):
    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-np.ones(n), np.ones(n-1)]
    offsets = [0, 1]
    L = sparse.diags(diagonals, offsets, shape=(n, n), format='lil')
    L /= h

    if bc == 'N':
        L[-1, -1] = 0
    
    return L.tocsr()

def grad_1d_backward(n, h, bc):
    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-np.ones(n-1), np.ones(n)]
    offsets = [-1, 0]
    L = sparse.diags(diagonals, offsets, shape=(n, n), format='lil')
    L /= h

    if bc == 'N':
        L[0, 0] = 0
    
    return L.tocsr()

def lap1d(N, dx, bc):
    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")
    
    diagonals = [np.ones(N-1), -2*np.ones(N), np.ones(N-1)]
    offsets = [-1, 0, 1]
    L = sparse.diags(diagonals, offsets, shape=(N, N), format='lil')
    L /= dx**2

    if bc == 'N':
        L[0, 0]   = -1 / dx**2
        L[0, 1]   = 1 / dx**2
        L[-1, -1] = -1 / dx**2
        L[-1, -2] = 1 / dx**2

    return L.tocsr()

def grad_staggered_grid(n, h, bc):
    if not bc in ['N']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-np.ones(n), np.ones(n)]
    offsets = [-1, 0]
    L = sparse.diags(diagonals, offsets, shape=(n+1, n), format='lil')
    L /= h

    if bc == 'N':
        L[0, 0]   = 0
        L[-1, -1] = 0

    return L.tocsr()

##################################### OPERATORS #####################################

def grad_st(Nt, Nx, Ny, dt, dx, dy, bc):
  Dt = grad_staggered_grid(Nt, dt, bc)
  Dx = grad_staggered_grid(Nx, dx, bc)
  Dy = grad_staggered_grid(Ny, dy, bc)

  Ixy = sparse.eye(Nx * Ny)
  It = sparse.eye(Nt)
  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  t = sparse.kron(Dt, Ixy)
  x = sparse.kron(sparse.eye(Nt), sparse.kron(Iy, Dx))
  y = sparse.kron(sparse.eye(Nt), sparse.kron(Dy, Ix))

  return bmat([ [t], [x], [y] ])

def div_st(Nt, Nx, Ny, dt, dx, dy, bc):
  Dt = grad_1d_central(Nt, dt, bc)
  Dx = grad_1d_central(Nx, dx, bc)
  Dy = grad_1d_central(Ny, dy, bc)

  Ixy = sparse.eye(Nx * Ny)
  It = sparse.eye(Nt)
  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  t = sparse.kron(Dt, Ixy)
  x = sparse.kron(It, sparse.kron(Iy, Dx))
  y = sparse.kron(It, sparse.kron(Dy, Ix))
  return bmat([ [t, x, y] ])

def laplacian_st(Nt, Nx, Ny, dt, dx, dy, bc):

    Lx = lap1d(Nx, dx, bc)
    Ly = lap1d(Ny, dy, bc)
    Lt = lap1d(Nt, dt, bc)
    Ix = sparse.eye(Nx)
    Iy = sparse.eye(Ny)
    It = sparse.eye(Nt)

    L_space = sparse.kron(Iy, Lx) + sparse.kron(Ly, Ix)
    I_space = sparse.eye(Nx*Ny)

    L_st = sparse.kron(Lt, I_space) + sparse.kron(It, L_space)
    return L_st


def grad(Nx, Ny, dx, dy, bc):
  Dx = grad_1d_central(Nx, dx, bc)
  Dy = grad_1d_central(Ny, dy, bc)

  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  x = sparse.kron(Iy, Dx)
  y = sparse.kron(Dy, Ix)
  return bmat([ [x], [y] ])

def div(Nx, Ny, dx, dy, bc):
  Dx = grad_1d_central(Nx, dx, bc)
  Dy = grad_1d_central(Ny, dy, bc)

  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  x = sparse.kron(Iy, Dx)
  y = sparse.kron(Dy, Ix)
  return bmat([ [x, y] ])