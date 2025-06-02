import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg

import benamou_brenier

Nx = 3
Ny = 3
Nt = 3

A = benamou_brenier.assemble_space_time_laplacian(Nt, 1, Nx, 1, Ny, 1)
print(A.todense())