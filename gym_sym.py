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


def compute_acceleration(D, D2, x, x_dot, boundary_val, g):

    # Compute required derivatives
    dx = D @ x
    d2x = D2 @ x
    dx_dot = D @ x_dot
    
    # Prepare tension eqn
    tension_lhs = D2 - np.diag(np.sum(d2x**2,axis=1))
    tension_rhs = -1 * np.sum(dx_dot * d2x,axis=1)

    # Prepare free end tension boundary condition
    tension_lhs[0,:] = np.zeros(len(tension_lhs))
    tension_lhs[0,0] = 1.0
    tension_rhs[0] = 0.0

    # Prepare fixed end boundary condition
    tension_lhs[-1,:] = D[-1,:]
    tension_rhs[-1] = np.dot(dx[-1], boundary_val) - np.dot(dx[-1], g)

    # Solve for tension
    tension = solve(tension_lhs, tension_rhs)

    # Compute acceleration (imposing boundary value)
    acceleration = (D @ tension).reshape((len(tension),1)) * dx + tension.reshape((len(tension),1)) * d2x + np.outer(np.ones(len(D)),g)
    acceleration[-1,:] = boundary_val

    return acceleration, tension


def test_acceleration():

    D, s = diff_matrix(10)
    D2 = D @ D
    x = np.column_stack((np.zeros(len(s)), -1 * s))
    x_dot = np.zeros(x.shape)
    boundary_val = np.zeros(2)
    g = np.array([0,-1.0])

    acceleration, tension = compute_acceleration(D, D2, x, x_dot, boundary_val, g)

    print(acceleration)
    print(tension)

    assert np.all(acceleration**2 < 10**(-5))


def time_step_jump(D, D2, g, x, x_dot, t, boundary_fn, step_size):
    '''Jump a time step using RK4'''

    k1x = x_dot
    k1xdot = compute_acceleration(D, D2, x, x_dot, boundary_fn(t), g)
    
    k2x = x_dot + 0.5 * step_size * k1xdot
    k2xdot = compute_acceleration(D, D2,
                                  x + 0.5 * step_size * k1x,
                                  x_dot + 0.5 * step_size * k1xdot,
                                  boundary_fn(t + 0.5 * step_size),
                                  g)

    k3x = x_dot + 0.5 * step_size * k2xdot
    k3xdot = compute_acceleration(D, D2,
                                  x + 0.5 * step_size * k2x,
                                  x_dot + 0.5 * step_size * k2xdot,
                                  boundary_fn(t + 0.5 * step_size),
                                  g)

    k4x = x_dot + step_size * k3xdot
    k4xdot = compute_acceleration(D, D2,
                                  x + step_size * k3x,
                                  x_dot + step_size * k3xdot,
                                  boundary_fn(t + step_size),
                                  g)

    new_x = x + (1/6) * step_size * (k1x + 2 * k2x + 2 * k3x + k4x)
    new_xdot = x_dot + (1/6) * step_size * (k1xdot + 2 * k2xdot + 2 * k3xdot + k4xdot)

    return new_x, new_xdot