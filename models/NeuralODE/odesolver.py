import math

def ode_solve(z0, t0, t1, f):
    """
    ODE 초기값 문제를 풀기 위한 가장 단순한 방법: Euler's Method
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z