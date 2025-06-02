import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg

dt = 1#./Nt
dx = 1#./Nx
dy = 1#./Ny

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

def space_time_div(v, Nt, Nx, Ny, dt, dx, dy):
    """
    Compute discrete divergence of a space-time vector field v = [v_t, v_x, v_y].

    Parameters:
        v : np.array, shape (3, Nt*Nx*Ny)
            Vector field components [time, x, y], each flattened.
        Nt, Nx, Ny : int
            Number of points in time and spatial dimensions.
        dt, dx, dy : float
            Grid spacings.

    Returns:
        div : np.array, shape (Nt*Nx*Ny,)
            Flattened scalar field of divergence at each point.
    """
    div = np.zeros(Nt*Nx*Ny)

    v_t, v_x, v_y = v

    def idx(t, y, x):
        return t*Nx*Ny + y*Nx + x

    for t in range(Nt):
        for y in range(Ny):
            for x in range(Nx):
                i = idx(t, y, x)

                # Time divergence (backward difference adjoint to forward gradient)
                if t == 0:
                    div[i] += -v_t[idx(t,y,x)] / dt
                elif t == Nt-1:
                    div[i] += v_t[idx(t-1,y,x)] / dt
                else:
                    div[i] += (v_t[idx(t-1,y,x)] - v_t[idx(t,y,x)]) / (2*dt)

                # X divergence
                if x == 0:
                    div[i] += -v_x[idx(t,y,x)] / dx
                elif x == Nx-1:
                    div[i] += v_x[idx(t,y,x-1)] / dx
                else:
                    div[i] += (v_x[idx(t,y,x-1)] - v_x[idx(t,y,x+1)]) / (2*dx)

                # Y divergence
                if y == 0:
                    div[i] += -v_y[idx(t,y,x)] / dy
                elif y == Ny-1:
                    div[i] += v_y[idx(t,y-1,x)] / dy
                else:
                    div[i] += (v_y[idx(t,y-1,x)] - v_y[idx(t,y+1,x)]) / (2*dy)

    return div


def space_time_div(mu, q, r, Nt, Nx, Ny, dt, dx, dy):
    """
    Compute discrete space-time divergence div_st(mu - r*q)
    Assume:
    - mu shape: (1, Nt*Nx*Ny)
    - q shape: (3, Nt*Nx*Ny), where q[0,:] = time component,
                                      q[1,:] = x component,
                                      q[2,:] = y component
    """
    div = np.zeros(Nt*Nx*Ny)

    # Extract flattened arrays for convenience
    mu_flat = mu[0]
    q_t = q[0]
    q_x = q[1]
    q_y = q[2]

    # Indices helper
    def idx(t, y, x):
        return t*Nx*Ny + y*Nx + x

    # Time divergence (forward difference at t=0, backward at t=Nt-1, centered inside)
    for t in range(Nt):
        for y in range(Ny):
            for x in range(Nx):
                i = idx(t,y,x)
                if t == 0:
                    dt_div = (mu_flat[idx(t+1,y,x)] - mu_flat[i]) / dt
                elif t == Nt-1:
                    dt_div = (mu_flat[i] - mu_flat[idx(t-1,y,x)]) / dt
                else:
                    dt_div = (mu_flat[idx(t+1,y,x)] - mu_flat[idx(t-1,y,x)]) / (2*dt)

                # Space divergence (central differences with Neumann BC)
                # x-direction
                if x == 0:
                    dx_div = (q_x[idx(t,y,x+1)] - q_x[i]) / dx
                elif x == Nx-1:
                    dx_div = (q_x[i] - q_x[idx(t,y,x-1)]) / dx
                else:
                    dx_div = (q_x[idx(t,y,x+1)] - q_x[idx(t,y,x-1)]) / (2*dx)

                # y-direction
                if y == 0:
                    dy_div = (q_y[idx(t,y+1,x)] - q_y[i]) / dy
                elif y == Ny-1:
                    dy_div = (q_y[i] - q_y[idx(t,y-1,x)]) / dy
                else:
                    dy_div = (q_y[idx(t,y+1,x)] - q_y[idx(t,y-1,x)]) / (2*dy)

                div[i] = dt_div + dx_div + dy_div

    return div

def solve_benamou_brenier_step(mu, q, rho0, rhoT, r, epsilon, Nt, Nx, Ny, dt, dx, dy):
    # Assemble operator
    L_st = assemble_space_time_laplacian(Nt, dt, Nx, dx, Ny, dy)
    I = sparse.eye(Nt*Nx*Ny)

    A = r * L_st + epsilon * I # L_st = -Lap

    # Compute RHS: divergence term
    F = space_time_div(mu, q, r, Nt, Nx, Ny, dt, dx, dy)

    # Add non-homogeneous Neumann BC correction in time at t=0 and t=Nt-1
    idx_t0 = np.arange(Nx*Ny)
    g0 = (rho0 - mu[0, 0:Nx*Ny] + r * q[0, 0:Nx*Ny]) / r
    F[idx_t0] += dt * g0

    idx_tN = np.arange(Nx*Ny) + (Nt-1)*Nx*Ny
    gN = (rhoT - mu[0, (Nt-1)*Nx*Ny : Nt*Nx*Ny] + r * q[0, (Nt-1)*Nx*Ny : Nt*Nx*Ny]) / r
    F[idx_tN] += dt * gN

    u = spsolve(A, F)
    # u, info = cg(A, F, rtol=1e-6, maxiter=1000)

    return u

def spaceTimeDiv(u, Nt, Nx, Ny, dt, dx, dy):
  gradTU = np.zeros(Nt*Nx*Ny)
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        if n > 0:
          gradTU[n*Nx*Ny+y*Nx+x] = (u[0, n*Nx*Ny+y*Nx+x] - u[0, (n-1)*Nx*Ny+y*Nx+x])
        else:
          gradTU[n*Nx*Ny+y*Nx+x] = (u[0, (n+1)*Nx*Ny+y*Nx+x] - u[0, n*Nx*Ny+y*Nx+x])

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

def spaceTimeGrad(u, Nt, Nx, Ny):
  gradTU = np.zeros(Nt*Nx*Ny)
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        if n < Nt-1:
          gradTU[n*Nx*Ny+y*Nx+x] = (u[(n+1)*Nx*Ny+y*Nx+x] - u[n*Nx*Ny+y*Nx+x])

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

def stepA(mu, q, rho0, rhoT, r, Nt, Nx, Ny):
  epsilon = 0.001

  spaceTimeDivSM = spaceTimeDiv(mu-r*q, Nt, Nx, Ny)

  F = np.zeros(Nt*Nx*Ny)
  Av = []
  Ax = []
  Ay = []
  for n in range(Nt):
    for y in range(Ny):
      for x in range(Nx):
        # Not on boundaries (space and time)
        if n>=1 and n<Nt-1 and y>=1 and y<Ny-1 and x>=1 and x<Nx-1:
          # A[n*Nx*Ny+y*Nx+x, (n+1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append((n+1)*Nx*Ny+y*Nx+x)
          Av.append(-r*1./dt**2)
          # A[n*Nx*Ny+y*Nx+x, (n-1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append((n-1)*Nx*Ny+y*Nx+x)
          Av.append(-r*1./dt**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+y*Nx+x+1)
          Av.append(-r*1./dx**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+y*Nx+x-1)
          Av.append(-r*1./dx**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+(y+1)*Nx+x)
          Av.append(-r*1./dy**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+(y-1)*Nx+x)
          Av.append(-r*1./dy**2)
          # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon
          Ax.append(n*Nx*Ny+y*Nx+x)
          Ay.append(n*Nx*Ny+y*Nx+x)
          Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon)

          F[n*Nx*Ny+y*Nx+x] = spaceTimeDivSM[n*Nx*Ny+y*Nx+x]
        else:
          # Neumann in time
          if n == 0:
            # A[n*Nx*Ny+y*Nx+x, (n+1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
            Ax.append(n*Nx*Ny+y*Nx+x)
            Ay.append((n+1)*Nx*Ny+y*Nx+x)
            Av.append(-r*1./dt**2)

            F[n*Nx*Ny+y*Nx+x] = spaceTimeDivSM[n*Nx*Ny+y*Nx+x] - 1/dt*(rho0[y*Nx+x]-mu[0,(0)*Nx*Ny+y*Nx+x]+r*q[0,(0)*Nx*Ny+y*Nx+x])

            # Not on space boundaries
            if y>=1 and y<Ny-1 and x>=1 and x<Nx-1:
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x+1)
              Av.append(-r*1./dx**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x-1)
              Av.append(-r*1./dx**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+(y+1)*Nx+x)
              Av.append(-r*1./dy**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+(y-1)*Nx+x)
              Av.append(-r*1./dy**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon +r*1./dt**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x)
              Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon - r*1./dt**2)
            else:
              if x == 0:
                if y == 0: # top-left corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2-r*1./dx**2 +r*1./dt**2 -r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2 -r*1./dx**2-r*1./dy**2) 
                elif y == Ny-1: # bottom-left corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2 +r*1./dt**2-r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2-r*1./dy**2)
                else: # on left but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2+r*1./dt**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dt**2-r*1./dx**2)
              elif x == Nx-1:
                if y == 0: # top-right corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2 -r*1./dy**2+r*1./dt**2-r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dt**2-r*1./dx**2-r*1./dy**2)
                elif y == Ny-1: # bottom-right corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2 -r*1./dy**2+r*1./dt**2-r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2-r*1./dy**2)
                else: # on right but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2+r*1./dt**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2)
              else:
                if y == 0: # on top but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2+r*1./dt**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dt**2-r*1./dy**2)
                elif y == Ny-1: # on bottom but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2+r*1./dt**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2)
          elif n == Nt-1:
            # A[n*Nx*Ny+y*Nx+x, (n-1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
            Ax.append(n*Nx*Ny+y*Nx+x)
            Ay.append((n-1)*Nx*Ny+y*Nx+x)
            Av.append(-r*1./dt**2)

            F[n*Nx*Ny+y*Nx+x] = spaceTimeDivSM[n*Nx*Ny+y*Nx+x ] + 1/dt*(rhoT[y*Nx+x]-mu[0,(Nt-1)*Nx*Ny+y*Nx+x]+r*q[0,(Nt-1)*Nx*Ny+y*Nx+x])
            # Not on space boundaries
            if y>=1 and y<Ny-1 and x>=1 and x<Nx-1:
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x+1)
              Av.append(-r*1./dx**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x-1)
              Av.append(-r*1./dx**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+(y+1)*Nx+x)
              Av.append(-r*1./dy**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+(y-1)*Nx+x)
              Av.append(-r*1./dy**2)
              # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon +r*1./dt**2
              Ax.append(n*Nx*Ny+y*Nx+x)
              Ay.append(n*Nx*Ny+y*Nx+x)
              Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2)
            else:
              if x == 0:
                if y == 0: # top-left corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2-r*1./dy**2+r*1./dt**2-r*1./dy**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2-r*1./dx**2)
                elif y == Ny-1: # bottom-left corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2 -r*1./dy**2+r*1./dt**2-r*1./dy**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dt**2-r*1./dy**2-r*1./dx**2)
                else: # on left but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2 +r*1./dt**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2)
              elif x == Nx-1:
                if y == 0: # top-right corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2+r*1./dt**2-r*1./dx**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2-r*1./dy**2)
                elif y == Ny-1: # bottom-right corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2+r*1./dt**2-r*1./dy**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2-r*1./dx**2)
                else: # on right but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2 +r*1./dt**2-r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dx**2)
              else:
                if y == 0: # on top but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 +r*1./dt**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2)
                elif y == Ny-1: # on bottom but not on corner
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x+1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x-1)
                  Av.append(-r*1./dx**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                  Av.append(-r*1./dy**2)
                  # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2 +r*1./dt**2-r*1./dy**2
                  Ax.append(n*Nx*Ny+y*Nx+x)
                  Ay.append(n*Nx*Ny+y*Nx+x)
                  Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dt**2-r*1./dy**2)
          else:
            # not on time boundaries
            F[n*Nx*Ny+y*Nx+x] = spaceTimeDivSM[n*Nx*Ny+y*Nx+x]
            # A[n*Nx*Ny+y*Nx+x, (n+1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
            Ax.append(n*Nx*Ny+y*Nx+x)
            Ay.append((n+1)*Nx*Ny+y*Nx+x)
            Av.append(-r*1./dt**2)
            # A[n*Nx*Ny+y*Nx+x, (n-1)*Nx*Ny+y*Nx+x] = -r*1./dt**2
            Ax.append(n*Nx*Ny+y*Nx+x)
            Ay.append((n-1)*Nx*Ny+y*Nx+x)
            Av.append(-r*1./dt**2)
            if x == 0:
              if y == 0: # top-left corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2-r*1./dx**2-r*1./dx**2-r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2-r*1./dy**2)
              elif y == Ny-1: # bottom-left corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2-r*1./dy**2-r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2-r*1./dx**2)
              else: # on left but not on corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2-r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2)
            elif x == Nx-1:
              if y == 0: # top-right corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2-r*1./dx**2-r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2-r*1./dy**2)
              elif y == Ny-1: # bottom-right corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2 -r*1./dx**2-r*1./dy**2-r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2-r*1./dx**2)
              else: # on right but not on corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dx**2-r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dx**2)
            else:
              if y == 0: # on top but not on corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y+1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y+1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2-r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2)
              elif y == Ny-1: # on bottom but not on corner
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x+1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x+1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x-1] = -r*1./dx**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x-1)
                Av.append(-r*1./dx**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+(y-1)*Nx+x] = -r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+(y-1)*Nx+x)
                Av.append(-r*1./dy**2)
                # A[n*Nx*Ny+y*Nx+x, n*Nx*Ny+y*Nx+x] = -r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon -r*1./dy**2-r*1./dy**2
                Ax.append(n*Nx*Ny+y*Nx+x)
                Ay.append(n*Nx*Ny+y*Nx+x)
                Av.append(-r*(-2./dt**2 - 2./dx**2 - 2./dy**2) + r*epsilon-r*1./dy**2)
  A = sparse.csr_matrix((Av, (Ax, Ay)), shape=[Nt*Nx*Ny, Nt*Nx*Ny])

# L = lap1d_neumann(N=100, dx=1.0/99)
# print(np.allclose(L.toarray(), L.T.toarray()))  # Should be True



# import numpy as np

# u = np.random.rand(Nt * Nx * Ny)
# v = np.random.rand(3, Nt * Nx * Ny)

# grad_u = space_time_grad(u, Nt, Nx, Ny, dt, dx, dy)
# div_v = space_time_div(v, Nt, Nx, Ny, dt, dx, dy)

# lhs = np.sum(grad_u * v)
# rhs = -np.sum(u * div_v)
# print(f"Adjoint test error: {np.abs(lhs - rhs)}")
# If that’s large (>1e-10), your discretizations are not compatible, and convergence will suffer or fail.

  return spsolve(A, F) 

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

def space_time_grad(u, Nt, Nx, Ny, dt, dx, dy):
    """
    Compute discrete gradient of u(t,y,x) with Neumann BCs.

    Parameters:
        u : array, shape (Nt*Nx*Ny,)
            Flattened scalar field on space-time grid.
        Nt, Nx, Ny : int
            Number of points in time and spatial dimensions.
        dt, dx, dy : float
            Grid spacings.

    Returns:
        grad : np.array, shape (3, Nt*Nx*Ny)
            Gradient components [time, x, y] each flattened.
    """
    grad_t = np.zeros_like(u)
    grad_x = np.zeros_like(u)
    grad_y = np.zeros_like(u)

    def idx(t, y, x):
        return t*Nx*Ny + y*Nx + x

    for t in range(Nt):
        for y in range(Ny):
            for x in range(Nx):
                i = idx(t,y,x)

                # Time gradient
                if t == 0:
                    grad_t[i] = (u[idx(t+1,y,x)] - u[i]) / dt
                elif t == Nt-1:
                    grad_t[i] = (u[i] - u[idx(t-1,y,x)]) / dt
                else:
                    grad_t[i] = (u[idx(t+1,y,x)] - u[idx(t-1,y,x)]) / (2*dt)

                # X gradient
                if x == 0:
                    grad_x[i] = (u[idx(t,y,x+1)] - u[i]) / dx
                elif x == Nx-1:
                    grad_x[i] = (u[i] - u[idx(t,y,x-1)]) / dx
                else:
                    grad_x[i] = (u[idx(t,y,x+1)] - u[idx(t,y,x-1)]) / (2*dx)

                # Y gradient
                if y == 0:
                    grad_y[i] = (u[idx(t,y+1,x)] - u[i]) / dy
                elif y == Ny-1:
                    grad_y[i] = (u[i] - u[idx(t,y-1,x)]) / dy
                else:
                    grad_y[i] = (u[idx(t,y+1,x)] - u[idx(t,y-1,x)]) / (2*dy)

    return np.array([grad_t, grad_x, grad_y])

# def solve(rho0, rhoT, Nt, Nx, Ny, r=1, epsilon=0.3, max_it=100):    
#     qPrev = np.zeros([3, Nt*Nx*Ny]) # = [a, b] \to\R^3
#     mu = np.zeros([3, Nt*Nx*Ny]) # = [rho, m] m\to \R^2
    
#     for i in range(0,max_it):
#       # phi = stepA(mu, qPrev, rho0, rhoT, r, Nt, Nx, Ny)
#       phi = solve_benamou_brenier_step(mu, qPrev, rho0, rhoT, r, epsilon, Nt, Nx, Ny, dt, dx, dy)
    
#       #spaceTimeGradPhi = spaceTimeGrad(phi, Nt, Nx, Ny)
#       spaceTimeGradPhi = space_time_grad(phi, Nt, Nx, Ny, dt, dx, dy)
#       q = stepB(spaceTimeGradPhi + 1./r*mu, Nt, Nx, Ny)
#       # stepC
#       muNext = mu + r*( spaceTimeGradPhi - q )
    
#       qPrev = q
#       mu = muNext
    
#       res = spaceTimeGradPhi[0,:] + 0.5*(np.power(spaceTimeGradPhi[1,:], 2) + np.power(spaceTimeGradPhi[2,:], 2) )
#       crit = np.sqrt( np.sum( np.multiply(mu[0,:], np.abs(res))) / np.sum( np.multiply(mu[0,:], np.power(spaceTimeGradPhi[1,:], 2) + np.power(spaceTimeGradPhi[2,:], 2) )) )
#       print(str(crit)+" ("+str(i+1)+"/"+str(max_it)+")")
    
#       if crit <= epsilon:
#         break
#     return mu, phi, q

def solve(rho0, rhoT, Nt, Nx, Ny, r=1, convergence_tol=0.3, reg_epsilon=1e-3, max_it=100):
    dt = 1. / (Nt - 1)
    dx = 1. / (Nx - 1)
    dy = 1. / (Ny - 1)

    qPrev = np.zeros([3, Nt*Nx*Ny])
    mu = np.zeros([3, Nt*Nx*Ny])
    crit = -1
    for i in range(max_it):
        phi = solve_benamou_brenier_step(mu, qPrev, rho0, rhoT, r, reg_epsilon, Nt, Nx, Ny, dt, dx, dy)

        gradPhi = space_time_grad(phi, Nt, Nx, Ny, dt, dx, dy)
        z = gradPhi + (1.0 / r) * mu
        q = stepB(z, Nt, Nx, Ny)

        mu = mu + r * (gradPhi - q)
        mu[0,:] = np.maximum(mu[0,:], 1e-10)  # Ensure positivity

        res = gradPhi[0,:] + 0.5 * (gradPhi[1,:]**2 + gradPhi[2,:]**2)
        num = np.sum(mu[0,:] * np.abs(res))
        denom = np.sum(mu[0,:] * (gradPhi[1,:]**2 + gradPhi[2,:]**2))

        prev_crit = crit
        crit = np.sqrt(num / (denom + 1e-10))  # prevent zero division
        print(f"{crit:.4e} ({i+1}/{max_it})")

        if crit <= convergence_tol:
            break
        
        if prev_crit >= 0:
          if np.abs(prev_crit - crit) < 1e-5:
            break

    return mu, phi, q