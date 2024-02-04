import gym_sym
import numpy as np

def time_step_search(ub=0.05, lb=0.0001, N=100):

    a = 0.5
    freq = 1.0
    
    position_fn = lambda t: np.array([a * np.sin(freq * t), 0])
    dposition_fn = lambda t: np.array([a * freq * np.cos(freq * t), 0])
    d2position_fn = lambda t: np.array([-1.0 * a * freq**2 * np.sin(freq * t), 0])
    boundary_fn = gym_sym.create_boundary_fn(position_fn, dposition_fn, d2position_fn)

    g = np.array([0,-5.0])
    x_initial_fn = lambda s: np.column_stack((np.zeros(len(s)), -1 * s))
    x_dot_initial_fn = lambda s: np.zeros((len(s),2))
    
    while True:
        step_size = np.sqrt(ub * lb)
        n_steps = int(10.0 / step_size)
        checkpoint_freq = int(0.1 / step_size)
        _, _, succeeded = gym_sym.evolve_with_disk_cache(boundary_fn, g, x_initial_fn, x_dot_initial_fn, N, step_size, n_steps, checkpoint_freq, end_mass=0.0, drag_coef=5.0)
        if succeeded:
            print(step_size, "succeeded")
            lb = step_size
        else:
            print(step_size, "failed")
            ub = step_size

        if lb > (9/10) * ub:
            break

    print(lb, ub)


def resolution_search(N=100, step_size=0.000389):

    a = 0.25
    freq = 6.0
    
    position_fn = lambda t: np.array([a * np.sin(freq * t), 0])
    dposition_fn = lambda t: np.array([a * freq * np.cos(freq * t), 0])
    d2position_fn = lambda t: np.array([-1.0 * a * freq**2 * np.sin(freq * t), 0])
    boundary_fn = gym_sym.create_boundary_fn(position_fn, dposition_fn, d2position_fn)

    g = np.array([0,-5.0])
    x_initial_fn = lambda s: np.column_stack((np.zeros(len(s)), -1 * s))
    x_dot_initial_fn = lambda s: np.zeros((len(s),2))

    while True:

        n_steps = int(10.0 / step_size)
        checkpoint_freq = int(0.1 / step_size)
        _, _, succeeded = gym_sym.evolve_with_disk_cache(boundary_fn, g, x_initial_fn, x_dot_initial_fn, N, step_size, n_steps, checkpoint_freq, end_mass=0.0, drag_coef=5.0)
        if succeeded:
            print(N, step_size, "Success!")
            break
        else:
            print(N, step_size, "Failure :(")
            step_size = step_size / 2
            N = int(N * 2**(1/2.5))