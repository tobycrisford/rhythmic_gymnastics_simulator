import numpy as np
from numpy.linalg import solve

def alt(n):
    alt = [(-1)**i for i in range(n)]
    return np.array(alt)

def diff_matrix(N):
    '''Stolen from here: https://hackmd.io/@NCTUIAM5804/Sk1JhoXoI'''

    if N==0:
        return 0, 1
    
    x = np.cos(np.pi*np.linspace(0,1,N+1))
    c = np.array([2] + [1]*(N-1)  + [2]) * alt(N+1)
    X = np.outer(x, np.ones(N+1))
    dX = X-X.T
    D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
    D = D - np.diag(np.sum(D,axis=1))
    return D, x


def compute_acceleration(D, D2, x, x_dot, boundary_val, dx, d2x, dx_dot, g):

    # Prepare tension eqn
    tension_lhs = D2 - np.diag(np.sum(d2x**2,axis=1))
    tension_rhs = -1 * np.sum(dx_dot * d2x,axis=1)

    # Prepare free end tension boundary condition
    tension_lhs[0,:] = np.zeros(len(tension_lhs))
    tension_lhs[0,0] = 1.0
    tenshion_rhs[0] = 0.0

    # Prepare fixed end boundary condition
    tension_lhs[-1,:] = D[-1,:]
    tension_rhs[-1] = np.dot(dx[-1], boundary_val)

    # Solve for tension
    tension = solve(tension_lhs, tension_rhs)

    # Compute acceleration (imposing boundary value)
    acceleration = (D @ tension) * dx + tension * d2x + np.outer(g * np.ones(len(D)))
    acceleration[-1,:] = boundary_val

    return acceleration