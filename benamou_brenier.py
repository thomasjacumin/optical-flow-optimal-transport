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
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg

def spaceTimeDiv(u, Nt, Nx, Ny, dt, dx, dy):
  """
    Compute the space-time divergence of a vector field `u`.

    Parameters
    ----------
    u : ndarray, shape (3, Nt*Nx*Ny)
        Input vector field components: time, x, and y components flattened over space-time.
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points in x-direction.
    Ny : int
        Number of spatial grid points in y-direction.
    dt : float
        Time step size.
    dx : float
        Spatial grid spacing in x-direction.
    dy : float
        Spatial grid spacing in y-direction.

    Returns
    -------
    ndarray, shape (Nt*Nx*Ny,)
        The divergence of `u` at each space-time point.
  """
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

def spaceTimeGrad(u, Nt, Nx, Ny, dt, dx, dy):
  """
    Compute the space-time gradient of a scalar field `u`.

    Parameters
    ----------
    u : ndarray, shape (Nt*Nx*Ny,)
        Input scalar field flattened over space-time.
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points in x-direction.
    Ny : int
        Number of spatial grid points in y-direction.
    dt : float
        Time step size.
    dx : float
        Spatial grid spacing in x-direction.
    dy : float
        Spatial grid spacing in y-direction.

    Returns
    -------
    ndarray, shape (3, Nt*Nx*Ny)
        The gradient components [time, x, y] at each space-time point.
  """
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

def lap1d_neumann(N, dx):
    """
    Construct a 1D Laplacian matrix with Neumann (zero-flux) boundary conditions.

    Parameters
    ----------
    N : int
        Number of grid points.
    dx : float
        Grid spacing.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse Laplacian operator matrix of size N x N.
    """
    diagonals = [
        np.ones(N-1),
        -2 * np.ones(N),
        np.ones(N-1)
    ]
    offsets = [-1, 0, 1]
    L = sparse.diags(diagonals, offsets, shape=(N, N), format='lil')
    L /= dx**2

    # Neumann BC: zero-flux at boundaries
    L[0, 0] = -1 / dx**2
    L[0, 1] = 1 / dx**2
    L[-1, -1] = -1 / dx**2
    L[-1, -2] = 1 / dx**2

    return L.tocsr()

def assemble_space_time_laplacian(Nt, dt, Nx, dx, Ny, dy):
    """
    Assemble the full space-time Laplacian operator combining time and 2D space.

    Parameters
    ----------
    Nt : int
        Number of time steps.
    dt : float
        Time step size.
    Nx : int
        Number of spatial grid points in x-direction.
    dx : float
        Spatial grid spacing in x-direction.
    Ny : int
        Number of spatial grid points in y-direction.
    dy : float
        Spatial grid spacing in y-direction.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix representing the combined space-time Laplacian.
    """
    Lx = lap1d_neumann(Nx, dx)
    Ly = lap1d_neumann(Ny, dy)
    Lt = lap1d_neumann(Nt, dt)
    Ix = sparse.eye(Nx)
    Iy = sparse.eye(Ny)
    It = sparse.eye(Nt)

    L_space = sparse.kron(Iy, Lx) + sparse.kron(Ly, Ix)
    I_space = sparse.eye(Nx*Ny)

    L_st = sparse.kron(Lt, I_space) + sparse.kron(It, L_space)
    return L_st

def solve_benamou_brenier_step(mu, q, rho0, rhoT, r, A, Nt, Nx, Ny, dt, dx, dy):
    """
    Perform one linear system solve step of the Benamou-Brenier formulation.

    Parameters
    ----------
    mu : ndarray, shape (3, Nt*Nx*Ny)
        Current iterate of density and momenta.
    q : ndarray, shape (3, Nt*Nx*Ny)
        Auxiliary momentum variables.
    rho0 : ndarray, shape (Nx*Ny,)
        Initial density distribution.
    rhoT : ndarray, shape (Nx*Ny,)
        Target density distribution.
    r : float
        Regularization parameter.
    A : scipy.sparse.csr_matrix
        Sparse system matrix.
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points in x-direction.
    Ny : int
        Number of spatial grid points in y-direction.
    dt : float
        Time step size.
    dx : float
        Spatial grid spacing in x-direction.
    dy : float
        Spatial grid spacing in y-direction.

    Returns
    -------
    ndarray, shape (Nt*Nx*Ny,)
        Solution vector for the step.
    """

    # Compute RHS: divergence term
    F = spaceTimeDiv(mu-r*q, Nt, Nx, Ny, dt, dx, dy)

    # Add non-homogeneous Neumann BC correction in time at t=0 and t=Nt-1
    idx_t0 = np.arange(Nx*Ny)
    g0 = -(rho0 - mu[0, 0:Nx*Ny] + r * q[0, 0:Nx*Ny])
    F[idx_t0] += 1/dt * g0

    idx_tN = np.arange(Nx*Ny) + (Nt-1)*Nx*Ny
    gN = (rhoT - mu[0, (Nt-1)*Nx*Ny : Nt*Nx*Ny] + r * q[0, (Nt-1)*Nx*Ny : Nt*Nx*Ny])
    F[idx_tN] += 1/dt * gN

    # u = spsolve(A, F)
    u, info = cg(A, F, rtol=1e-6, maxiter=1000)

    return u

def stepB(p, Nt, Nx, Ny):
  """
    Nonlinear projection step applying cubic root computations.

    Parameters
    ----------
    p : ndarray, shape (3, Nt*Nx*Ny)
        Input variables consisting of alpha and beta components.
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points in x-direction.
    Ny : int
        Number of spatial grid points in y-direction.

    Returns
    -------
    ndarray, shape (3, Nt*Nx*Ny)
        Projected variables after nonlinear transformation.
  """
  
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

def solve(rho0, rhoT, Nt, Nx, Ny, r=1, convergence_tol=0.3, reg_epsilon=1e-3, max_it=100):
    """
    Solve the optimal transport problem using the Benamou-Brenier formulation
    with iterative updates.

    Parameters
    ----------
    rho0 : ndarray, shape (Nx*Ny,)
        Initial density distribution.
    rhoT : ndarray, shape (Nx*Ny,)
        Final density distribution.
    Nt : int
        Number of time steps.
    Nx : int
        Number of spatial grid points in x-direction.
    Ny : int
        Number of spatial grid points in y-direction.
    r : float, optional
        Regularization parameter (default is 1).
    convergence_tol : float, optional
        Convergence tolerance for iterative solver (default is 0.3).
    reg_epsilon : float, optional
        Regularization epsilon added to system matrix diagonal (default is 1e-3).
    max_it : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    tuple of ndarray
        - mu: ndarray, shape (3, Nt*Nx*Ny), final density and momenta.
        - phi: ndarray, shape (Nt*Nx*Ny,), potential field.
        - q: ndarray, shape (3, Nt*Nx*Ny), auxiliary momentum variables.
    """

    dt = 1
    dx = 1
    dy = 1

    qPrev = np.zeros([3, Nt*Nx*Ny]) # = [a, b] \to\R^3
    mu = np.zeros([3, Nt*Nx*Ny]) # = [rho, m] m\to \R^2
    crit = -1
    # Assemble operator
    L_st = assemble_space_time_laplacian(Nt, dt, Nx, dx, Ny, dy)
    I = sparse.eye(Nt*Nx*Ny)
    A = -r*L_st + reg_epsilon*I
    for i in range(0,max_it):
      # stepA
      phi = solve_benamou_brenier_step(mu, qPrev, rho0, rhoT, r, A, Nt, Nx, Ny, dt, dx, dy)
      # stepB
      gradPhi = spaceTimeGrad(phi, Nt, Nx, Ny, dt, dx, dy)
      q = stepB(gradPhi + (1./r)*mu, Nt, Nx, Ny)
      # stepC
      mu = mu + r*( gradPhi - q )
      mu[0,:] = np.maximum(mu[0,:], 1e-10)  # Ensure positivity
      qPrev = q
      
      res = gradPhi[0,:] + 0.5 * (gradPhi[1,:]**2 + gradPhi[2,:]**2)
      num = np.sum(mu[0,:] * np.abs(res))
      denom = np.sum(mu[0,:] * (gradPhi[1,:]**2 + gradPhi[2,:]**2))

      prev_crit = crit
      crit = np.sqrt(num / (denom + 1e-10))  # prevent zero division
      print(str(crit)+" ("+str(i+1)+"/"+str(max_it)+")")
    
      if crit <= convergence_tol:
        break
      if prev_crit >= 0:
        if np.abs(prev_crit - crit) < 1e-5:
          break

    return mu, phi, q