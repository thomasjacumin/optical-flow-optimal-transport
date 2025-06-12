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

import operators

from PIL import Image
import utils

def solve_benamou_brenier_step(mu, q, rho0, rhoT, r, A, div, Nt, Nx, Ny, dt, dx, dy):
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
    F = div@(mu-r*q)

    # for n in range(0,Nt):
    #     Fn = F[n*Nx*Ny:(n+1)*Nx*Ny]
    #     print(f"Fn-{n} - min: {np.min(Fn)} max: {np.max(Fn)}")
    #     Fn  = (Fn-np.min(Fn))/(np.max(Fn)-np.min(Fn))
    #     Image.fromarray(np.uint8(255*np.clip(Fn, 0, 1).reshape([Ny,Nx])), 'L').save(f"results/Fn-{n}.png")

    # Add non-homogeneous Neumann BC correction in time at t=0 and t=Nt-1
    rho = mu[0:Nt*Nx*Ny]
    a = q[0:Nt*Nx*Ny]
    idx_t0 = np.arange(Nx*Ny)
    # g0 = rho0 - mu[0, 0:Nx*Ny] + r * q[0, 0:Nx*Ny]
    g0 = rho0 - rho[0:Nx*Ny] + r*a[0:Nx*Ny]
    F[idx_t0] -= 1/dt * g0

    idx_tN = np.arange(Nx*Ny) + (Nt-1)*Nx*Ny
    gN = rhoT - rho[(Nt-1)*Nx*Ny:Nt*Nx*Ny] + r*a[(Nt-1)*Nx*Ny:Nt*Nx*Ny]
    F[idx_tN] += 1/dt * gN

    # u = spsolve(A, F)
    u, info = cg(A, F, rtol=1e-6, maxiter=1000)
    if info > 0:
        print(f"WARNING: CG did not converge in {info} iterations.")
    elif info < 0:
        raise RuntimeError("CG solver failed due to illegal input or breakdown.")

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
    alpha = p[i]
    beta1 = p[Nt*Nx*Ny + i]
    beta2 = p[2*Nt*Nx*Ny + i]

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
  return np.concatenate((a, b1, b2))

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

    # qPrev = np.zeros([3, Nt*Nx*Ny]) # = [a, b] \to\R^3
    # mu = np.zeros([3, Nt*Nx*Ny]) # = [rho, m] \to \R^2
    qPrev = np.zeros([3*Nt*Nx*Ny]) # = [a, b1, b2]
    mu = np.zeros([3*Nt*Nx*Ny]) # = [rho, m1, m2]
    for n in range(Nt):
        mu[n*Nx*Ny:(n+1)*Nx*Ny] = (1 - n / (Nt - 1)) * rho0 + (n / (Nt - 1)) * rhoT
    crit = -1
    # Assemble operator
    grad_st = operators.grad_st(Nt, Nx, Ny, dt, dx, dy, bc='N')
    # div_st  = -grad_st.transpose()
    div_st  = operators.div_st(Nt, Nx, Ny, dt, dx, dy, bc='N')
    # L_st    = div_st@grad_st
    L_st    = operators.laplacian_st(Nt, Nx, Ny, dt, dx, dy, bc='N')
    I = sparse.eye(Nt*Nx*Ny)
    A = -r*L_st + r*reg_epsilon*I
    for i in range(0, max_it):
      # stepA
      phi = solve_benamou_brenier_step(mu, qPrev, rho0, rhoT, r, A, div_st, Nt, Nx, Ny, dt, dx, dy)
    #   for n in range(0,Nt):
    #     phin = phi[n*Nx*Ny:(n+1)*Nx*Ny]
    #     print(f"phin-{n} - min: {np.min(phin)} max: {np.max(phin)}")
    #     phin  = (phin-np.min(phin))/(np.max(phin)-np.min(phin))
    #     Image.fromarray(np.uint8(255*np.clip(phin, 0, 1).reshape([Ny,Nx])), 'L').save(f"results/phi-{n}.png")
      # stepB
      gradPhi = (grad_st@phi) # spaceTimeGrad(phi, Nt, Nx, Ny, dt, dx, dy)
      q = stepB(gradPhi + (1./r)*mu, Nt, Nx, Ny)
    #   for n in range(0,Nt):
    #     an  = q[n*Nx*Ny:(n+1)*Nx*Ny]
    #     print(f"an-{n} - min: {np.min(an)} max: {np.max(an)}")
    #     an  = (an-np.min(an))/(np.max(an)-np.min(an))
    #     b1n = q[Nt*Nx*Ny + n*Nx*Ny:Nt*Nx*Ny + (n+1)*Nx*Ny]
    #     print(f"b1n-{n} - min: {np.min(b1n)} max: {np.max(b1n)}")
    #     b1n  = (b1n-np.min(b1n))/(np.max(b1n)-np.min(b1n))
    #     b2n = q[2*Nt*Nx*Ny + n*Nx*Ny:2*Nt*Nx*Ny + (n+1)*Nx*Ny]
    #     print(f"b2n-{n} - min: {np.min(b2n)} max: {np.max(b2n)}")
    #     b2n  = (b2n-np.min(b2n))/(np.max(b2n)-np.min(b2n))
    #     Image.fromarray(np.uint8(255*np.clip(an, 0, 1).reshape([Ny,Nx])), 'L').save(f"results/a-{n}.png")
    #     Image.fromarray(np.uint8(255*np.clip(b1n, 0, 1).reshape([Ny,Nx])), 'L').save(f"results/b1-{n}.png")
    #     Image.fromarray(np.uint8(255*np.clip(b2n, 0, 1).reshape([Ny,Nx])), 'L').save(f"results/b2-{n}.png")
      # stepC
      mu = mu + r*( gradPhi - q )

      # mu[0,:] = np.maximum(mu[0,:], 1e-10)  # Ensure positivity
      mu[0:Nt*Nx*Ny] = np.maximum(mu[0:Nt*Nx*Ny], 0)  # Ensure positivity
      # After mu update in main loop:
    #   mu[0:Nt*Nx*Ny][:Nx*Ny] = rho0
    #   mu[0:Nt*Nx*Ny][-Nx*Ny:] = rhoT
      qPrev = q

    #   print(f"Iter {i}: mu min {mu[0:Nt*Nx*Ny].min()}, max {mu[0:Nt*Nx*Ny].max()}, sum {mu[0:Nt*Nx*Ny].sum()}")
    #   print("norm gradPhi:", np.linalg.norm(gradPhi))
    #   print("norm q:", np.linalg.norm(q))
      
      # res = gradPhi[0,:] + 0.5 * (gradPhi[1,:]**2 + gradPhi[2,:]**2)
      # num = np.sum(mu[0,:] * np.abs(res))
      # denom = np.sum(mu[0,:] * (gradPhi[1,:]**2 + gradPhi[2,:]**2))

      res = gradPhi[0:Nt*Nx*Ny] + 0.5 * (gradPhi[Nt*Nx*Ny:2*Nt*Nx*Ny]**2 + gradPhi[2*Nt*Nx*Ny:3*Nt*Nx*Ny]**2)
      num = np.sum(mu[0:Nt*Nx*Ny] * np.abs(res))
      denom = np.sum(mu[0:Nt*Nx*Ny] * (gradPhi[Nt*Nx*Ny:2*Nt*Nx*Ny]**2 + gradPhi[2*Nt*Nx*Ny:3*Nt*Nx*Ny]**2))

      prev_crit = crit
      crit = np.sqrt(num / (denom + 1e-10))  # prevent zero division
      print(str(crit)+" ("+str(i+1)+"/"+str(max_it)+")")

      if crit <= convergence_tol:
        break
      if prev_crit >= 0:
        if np.abs(prev_crit - crit) < 1e-5:
          break

    Image.fromarray(np.uint8(255*np.clip(Nx*Ny*rho0, 0, 1).reshape([Ny,Nx])), 'L').save("results/rho0.png")
    Image.fromarray(np.uint8(255*np.clip(Nx*Ny*rhoT, 0, 1).reshape([Ny,Nx])), 'L').save("results/rhoT.png")
    rhon = np.zeros([Nt, Nx*Ny])
    for n in range(0, Nt):
      rhon[n,:] = mu[n*Nx*Ny:(n+1)*Nx*Ny]
      IE = utils.IE(Nx, Ny, rhoT, rhon[n,:])
      print(f"{n}: mass:{np.sum(rhon[n,:])}")
      Image.fromarray(np.uint8(255*np.clip(Nx*Ny*rhon[n,:], 0, 1).reshape([Ny,Nx])), 'L').save(f"results/{n}.png")

    grad = operators.grad(Nx, Ny, dx, dy, bc='N')
    div  = operators.div(Nx, Ny, dx, dy, bc='D')
    return utils.opticalflow_from_benamoubrenier(phi, Nt, Nx, Ny, grad, div)