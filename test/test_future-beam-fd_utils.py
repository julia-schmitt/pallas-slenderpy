import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD


def test_first_derivative():
    ### y'(x) = sin(x) on [-1,2] with y(-1) = 3 ###

    left_bound = -1
    right_bound = 2
    n = 10000
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)

    rhs = np.sin(x)
    rhs[0] = 3

    bc_matrix = sp.sparse.lil_matrix((n, n))
    bc_matrix[0, 0] = 1
    bc_matrix[-1, -1] = 1 / ds
    bc_matrix[-1, -2] = -1 / ds

    A = FD.first_derivative(n, ds)

    sol = sp.sparse.linalg.spsolve(A + bc_matrix, rhs)

    atol = 1.0e-06
    rtol = 1.0e-09

    def exact(x):
        return -np.cos(x) + 3 + np.cos(-1)

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_second_derivative():
    ### y"(x) = 0 on [0,1]  with y(0) = 0 and y'(1) = 2 ###

    left_bound = 0
    right_bound = 1
    n = 100
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)

    rhs = np.zeros(n)
    rhs[0] = 1
    rhs[-1] = 2

    bc_matrix = sp.sparse.lil_matrix((n, n))
    bc_matrix[0, 0] = 1
    bc_matrix[-1, -1] = 1 / ds
    bc_matrix[-1, -2] = -1 / ds

    A = FD.second_derivative(n, ds)

    sol = sp.sparse.linalg.spsolve(A + bc_matrix, rhs)

    atol = 1.0e-06
    rtol = 1.0e-09

    def exact(x):
        return 2 * x + 1

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_boundary_condition_order2():
    ### y"(x) = 0 on [0,1]  with y(0) = 0 and y'(1) = 2 ###

    left_bound = 0
    right_bound = 1
    n = 100
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)
    order = 2

    left = [[1, 0, 0, 1]]
    right = [[0, 1, 0, 2]]
    bc = FD.BoundaryCondition(order, left, right)
    BC, rhs = bc.compute(ds, n)
    A = FD.second_derivative(n, ds)

    sol = sp.sparse.linalg.spsolve(A + BC, rhs)

    atol = 1.0e-06
    rtol = 1.0e-09

    def exact(x):
        return 2 * x + 1

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_fourth_derivative():
    ### y""(x) = 0 on [0,1]  with y(0) = 0 and y'(0) = 1 ###
    ###                           y(1) = 1 and y'(1) = 2 ###
    left_bound = 0
    right_bound = 1
    n = 10000
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)

    rhs = np.zeros(n)
    rhs[0] = 0
    rhs[1] = 1
    rhs[-2] = 1
    rhs[-1] = 2

    bc_matrix = sp.sparse.lil_matrix((n, n))
    bc_matrix[0, 0] = 1
    bc_matrix[1, 1] = 1 / ds
    bc_matrix[1, 0] = -1 / ds
    bc_matrix[-2, -2] = 1
    bc_matrix[-1, -1] = 1 / ds
    bc_matrix[-1, -2] = -1 / ds

    A = FD.fourth_derivative(n, ds)

    sol = sp.sparse.linalg.spsolve(A + bc_matrix, rhs)

    atol = 1.0e-06
    rtol = 1.0e-03

    def exact(x):
        return x**3 - x**2 + x

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)


def test_boundary_condition_order4():
    ### y""(x) = 0 on [0,1]  with y(0) = 0 and y'(0) = 1 ###
    ###                           y(1) = 1 and y'(1) = 2 ###

    left_bound = 0
    right_bound = 1
    n = 10000
    ds = (right_bound - left_bound) / (n - 1)
    x = np.linspace(left_bound, right_bound, n)
    order = 4

    left = [[1, 0, 0, 0], [0, 1, 0, 1]]
    right = [[1, 0, 0, 1], [0, 1, 0, 2]]
    bc = FD.BoundaryCondition(order, left, right)
    BC, rhs = bc.compute(ds, n)
    A = FD.fourth_derivative(n, ds)

    sol = sp.sparse.linalg.spsolve(A + BC, rhs)

    atol = 1.0e-04
    rtol = 1.0e-04

    def exact(x):
        return x**3 - x**2 + x

    assert np.allclose(exact(x), sol, atol=atol, rtol=rtol)
