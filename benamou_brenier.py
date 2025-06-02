import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

def spaceTimeDiv(u, Nt, Nx, Ny, dt, dx, dy):
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

def solve_benamou_brenier_step(mu, q, rho0, rhoT, r, epsilon, Nt, Nx, Ny, dt, dx, dy):
    # Assemble operator
    L_st = assemble_space_time_laplacian(Nt, dt, Nx, dx, Ny, dy)
    I = sparse.eye(Nt*Nx*Ny)

    A = -r * L_st + epsilon * I

    # Compute RHS: divergence term
    F = spaceTimeDiv(mu-r*q, Nt, Nx, Ny, dt, dx, dy)

    # Add non-homogeneous Neumann BC correction in time at t=0 and t=Nt-1
    idx_t0 = np.arange(Nx*Ny)
    g0 = -(rho0 - mu[0, 0:Nx*Ny] + r * q[0, 0:Nx*Ny])
    F[idx_t0] += 1/dt * g0

    idx_tN = np.arange(Nx*Ny) + (Nt-1)*Nx*Ny
    gN = (rhoT - mu[0, (Nt-1)*Nx*Ny : Nt*Nx*Ny] + r * q[0, (Nt-1)*Nx*Ny : Nt*Nx*Ny])
    F[idx_tN] += 1/dt * gN

    u = spsolve(A, F)

    return u

def stepB(p, Nt, Nx, Ny):
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
    dt = 1
    dx = 1
    dy = 1

    qPrev = np.zeros([3, Nt*Nx*Ny]) # = [a, b] \to\R^3
    mu = np.zeros([3, Nt*Nx*Ny]) # = [rho, m] m\to \R^2
    crit = -1
    for i in range(0,max_it):
      phi = solve_benamou_brenier_step(mu, qPrev, rho0, rhoT, r, reg_epsilon, Nt, Nx, Ny, dt, dx, dy)
    
      gradPhi = spaceTimeGrad(phi, Nt, Nx, Ny, dt, dx, dy)
      q = stepB(gradPhi + (1./r)*mu, Nt, Nx, Ny)
      # stepC
      muNext = mu + r*( gradPhi - q )
      mu[0,:] = np.maximum(mu[0,:], 1e-10)  # Ensure positivity
    
      qPrev = q
      mu = muNext

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