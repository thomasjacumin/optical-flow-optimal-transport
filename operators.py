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

from scipy import sparse
from scipy.sparse import bmat
import numpy as np

##################################### FD Schemes #####################################

def grad_1d_central(n, h, bc):
    """
    Constructs a 1D central difference gradient matrix with custom boundary handling.

    Parameters
    ----------
    n : int
        Number of grid points in the 1D domain.
    h : float
        Grid spacing.
    bc : str
        Boundary condition type. Must be either:
        - 'N' for Neumann
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape (n, n) representing the central finite difference
        approximation of the gradient operator.

    Raises
    ------
    NotImplementedError
        If `bc` is not 'N' or 'D'.
    """

    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-0.5*np.ones(n-1), 0.5*np.ones(n-1)]
    offsets = [-1, 1]
    L = sparse.diags(diagonals, offsets, shape=(n, n), format='lil')
    L /= h

    if bc == 'N':
        L[0, 1]   = 0
        L[-1, -2] = 0
    
    return L.tocsr()

def grad_1d_central_weird(n, h, bc):
    """
    Constructs a 1D central difference gradient matrix with custom boundary handling.

    Parameters
    ----------
    n : int
        Number of grid points in the 1D domain.
    h : float
        Grid spacing.
    bc : str
        Boundary condition type. Must be either:
        - 'N' for Neumann
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape (n, n) representing the central finite difference
        approximation of the gradient operator.

    Raises
    ------
    NotImplementedError
        If `bc` is not 'N' or 'D'.
    """
        
    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-0.5*np.ones(n-1), 0.5*np.ones(n-1)]
    offsets = [-1, 1]
    L = sparse.diags(diagonals, offsets, shape=(n, n), format='lil')
    L /= h

    if bc == 'N':
        L[0, 0]   = -1
        L[0, 1]   = 1
        L[-1, -1] = 1
        L[-1, -2] = -1
    
    return L.tocsr()

def grad_1d_forward(n, h, bc):
    """
    Constructs a 1D forward difference gradient matrix with custom boundary handling.

    Parameters
    ----------
    n : int
        Number of grid points in the 1D domain.
    h : float
        Grid spacing.
    bc : str
        Boundary condition type. Must be either:
        - 'N' for Neumann
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape (n, n) representing the forward finite difference
        approximation of the gradient operator.

    Raises
    ------
    NotImplementedError
        If `bc` is not 'N' or 'D'.
    """
    if not bc in ['N', 'D']:
        raise NotImplementedError("These boundary conditions are not implemented")

    diagonals = [-np.ones(n), np.ones(n-1)]
    offsets = [0, 1]
    L = sparse.diags(diagonals, offsets, shape=(n, n), format='lil')
    L /= h

    if bc == 'N':
        L[-1, -1] = 0
    
    return L.tocsr()

def lap1d(N, dx, bc):
    """
    Constructs a 1D Laplacian operator with finite differences and specified boundary conditions.

    Parameters
    ----------
    N : int
        Number of grid points in the 1D domain.
    dx : float
        Grid spacing.
    bc : str
        Boundary condition type. Must be either:
        - 'N' for Neumann boundary conditions (zero-flux at boundaries)
        - 'D' for Dirichlet boundary conditions (default: values at boundaries)

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape (N, N) representing the finite-difference approximation
        of the 1D Laplace operator.

    Raises
    ------
    NotImplementedError
        If `bc` is not 'N' or 'D'.
    """
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

##################################### OPERATORS #####################################

def grad_st(Nt, Nx, Ny, dt, dx, dy, bc):
  """
    Constructs a space-time gradient operator for a 3D domain (time, x, y) using 
    central finite differences.

    Parameters
    ----------
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points in the x-direction.
    Ny : int
        Number of spatial grid points in the y-direction.
    dt : float
        Time step size.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    bc : str
        Boundary condition type, passed to the 1D gradient constructor.
        Must be either:
        - 'N' for Neumann
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.bmat
        A block sparse matrix representing the space-time gradient operator
        with three stacked components: time, x, and y derivatives.

    Raises
    ------
    Any error raised by `grad_1d_central_weird`, typically a NotImplementedError
    if the boundary condition type is not supported.
    """
  Dt = grad_1d_central_weird(Nt, dt, bc)
  Dx = grad_1d_central_weird(Nx, dx, bc)
  Dy = grad_1d_central_weird(Ny, dy, bc)

  Ixy = sparse.eye(Nx * Ny)
  It = sparse.eye(Nt)
  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  t = sparse.kron(Dt, Ixy)
  x = sparse.kron(It, sparse.kron(Iy, Dx))
  y = sparse.kron(It, sparse.kron(Dy, Ix))
  return bmat([ [t], [x], [y] ])

def div_st(Nt, Nx, Ny, dt, dx, dy, bc):
  """
    Constructs a space-time divergence operator for a 3D domain (time, x, y) using 
    central finite differences.

    Parameters
    ----------
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points in the x-direction.
    Ny : int
        Number of spatial grid points in the y-direction.
    dt : float
        Time step size.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    bc : str
        Boundary condition type, passed to the 1D gradient constructor.
        Must be either:
        - 'N' for Neumann
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.bmat
        A block sparse matrix representing the space-time divergence operator
        with three stacked components: time, x, and y derivatives.

    Raises
    ------
    Any error raised by `grad_1d_central_weird`, typically a NotImplementedError
    if the boundary condition type is not supported.
    """
  Dt = grad_1d_central_weird(Nt, dt, bc)
  Dx = grad_1d_central_weird(Nx, dx, bc)
  Dy = grad_1d_central_weird(Ny, dy, bc)

  Ixy = sparse.eye(Nx * Ny)
  It = sparse.eye(Nt)
  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  t = sparse.kron(Dt, Ixy)
  x = sparse.kron(It, sparse.kron(Iy, Dx))
  y = sparse.kron(It, sparse.kron(Dy, Ix))
  return bmat([ [t, x, y] ])

def laplacian_st(Nt, Nx, Ny, dt, dx, dy, bc):
    """
    Constructs a 3D space-time Laplacian operator using finite differences
    over a regular grid with specified boundary conditions.

    Parameters
    ----------
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points in the x-direction.
    Ny : int
        Number of spatial grid points in the y-direction.
    dt : float
        Time step size.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    bc : str
        Boundary condition type for all dimensions.
        Must be either:
        - 'N' for Neumann (zero-flux)
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape (Nt * Nx * Ny, Nt * Nx * Ny) representing the
        3D Laplacian operator in space and time.
    Raises
    ------
    NotImplementedError
        If `bc` is not one of the supported boundary conditions.
    """
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
  """
    Constructs a space gradient operator for a 2D domain (x, y) using 
    central finite differences.

    Parameters
    ----------
    Nx : int
        Number of spatial grid points in the x-direction.
    Ny : int
        Number of spatial grid points in the y-direction.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    bc : str
        Boundary condition type, passed to the 1D gradient constructor.
        Must be either:
        - 'N' for Neumann
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.bmat
        A block sparse matrix representing the space gradient operator
        with three stacked components: x and y derivatives.

    Raises
    ------
    Any error raised by `grad_1d_central`, typically a NotImplementedError
    if the boundary condition type is not supported.
    """
  Dx = grad_1d_central(Nx, dx, bc)
  Dy = grad_1d_central(Ny, dy, bc)

  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  x = sparse.kron(Iy, Dx)
  y = sparse.kron(Dy, Ix)
  return bmat([ [x], [y] ])

def grad_forward(Nx, Ny, dx, dy, bc='N'):
  """
    Constructs a space gradient operator for a 2D domain (x, y) using 
    forward finite differences.

    Parameters
    ----------
    Nx : int
        Number of spatial grid points in the x-direction.
    Ny : int
        Number of spatial grid points in the y-direction.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    bc : str
        Boundary condition type, passed to the 1D gradient constructor.
        Must be either:
        - 'N' for Neumann
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.bmat
        A block sparse matrix representing the space gradient operator
        with three stacked components: x and y derivatives.

    Raises
    ------
    Any error raised by `grad_1d_forward`, typically a NotImplementedError
    if the boundary condition type is not supported.
    """
  Dx = grad_1d_forward(Nx, dx, bc)
  Dy = grad_1d_forward(Ny, dy, bc)

  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  x = sparse.kron(Iy, Dx)
  y = sparse.kron(Dy, Ix)
  return bmat([ [x], [y] ])

def div(Nx, Ny, dx, dy, bc):
  """
    Constructs a space divergence operator for a 2D domain (x, y) using 
    central finite differences.

    Parameters
    ----------
    Nx : int
        Number of spatial grid points in the x-direction.
    Ny : int
        Number of spatial grid points in the y-direction.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.
    bc : str
        Boundary condition type, passed to the 1D gradient constructor.
        Must be either:
        - 'N' for Neumann
        - 'D' for Dirichlet

    Returns
    -------
    scipy.sparse.bmat
        A block sparse matrix representing the space divergence operator
        with three stacked components: x and y derivatives.

    Raises
    ------
    Any error raised by `grad_1d_central`, typically a NotImplementedError
    if the boundary condition type is not supported.
    """
  Dx = grad_1d_central(Nx, dx, bc)
  Dy = grad_1d_central(Ny, dy, bc)

  Ix = sparse.eye(Nx)
  Iy = sparse.eye(Ny)

  x = sparse.kron(Iy, Dx)
  y = sparse.kron(Dy, Ix)
  return bmat([ [x, y] ])