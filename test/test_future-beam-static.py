import numpy as np

from slenderpy.future.beam.static import _solve_curvature_approx
from slenderpy.future.beam.fd_utils import BoundaryCondition


def test_solve_cruvature_approx_order2():
    ### y"(x) = x on [2,3]  with y(2) -y(2) + 3y"(2) = 0###
    ###                      and -y(3) + y"(3) = 4 ###

    left_bound = 2
    right_bound = 3
    n = 10000
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)

    EI = 0
    H = -1

    left = [[1, -1, 3, 0]]
    right = [[-1, 0, 1, 4]]
    order = 2
    bc = BoundaryCondition(order, left, right)
    sol = _solve_curvature_approx(n, ds, EI, H, bc, rhs=np.copy(x))

    def exact(x):
        A = -1 / 12
        B = -63 / 12
        return x**3 / 6 + A * x + B

    atol = 1.0e-09
    rtol = 1.0e-04

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_solve_cruvature_approx_order4():
    ### solve y"" - y" = 0 on [0,1] with y(0) = 0  y"(0) = 1 ###
    ###                              and y(1) = 0  y"(1) = 0 ###
    left_bound = 0
    right_bound = 1
    n = 10000
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)

    EI = 1
    H = 1

    left = [[1, 0, 0, 0], [0, 0, 1, 1]]
    right = [[1, 0, 0, 0], [0, 0, 1, 0]]
    bc = BoundaryCondition(4, left, right)
    sol = _solve_curvature_approx(n, ds, EI, H, bc)

    def exact(x):
        A = -1 / (np.exp(1) ** 2 - 1)
        B = np.exp(1) ** 2 / (np.exp(1) ** 2 - 1)
        D = -B - A
        C = -D - A * np.exp(1) - B * np.exp(-1)
        return A * np.exp(x) + B * np.exp(-x) + C * x + D

    atol = 1.0e-05
    rtol = 1.0e-09

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)
