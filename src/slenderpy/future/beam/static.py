from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
from slenderpy.future._constant import _GRAVITY


def clean_matrix(order: int, A2: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
    """Earase the proper coefficient in the scheme matrix to take into account the bounday conditions."""
    if order == 4:
        A2.data[0, 0] = 0
        A2.data[0, -3] = 0

        A2.data[1, 1] = 0
        A2.data[1, -2] = 0

        A2.data[2, -1] = 0
        A2.data[2, 2] = 0

    return A2


def clean_rhs(order: int, rhs: Optional[np.ndarray[float]] = None) -> np.ndarray[float]:
    """Earase the proper coefficient in the right hand-side to take into account the bounday conditions."""
    rhs[0] = 0
    rhs[-1] = 0

    if order == 4:
        rhs[1] = 0
        rhs[-2] = 0

    return rhs


def _solve_curvature_approx(
    n: int,
    bc: FD.BoundaryCondition,
    lspan: float = 400.0,
    tension: float = 1.853e04,
    ei: float = 2155.0,
    rhs: Optional[np.ndarray[float]] = None,
) -> Optional[np.ndarray[float]]:
    """Solve equation of the form : ei*(d^4/dx^4)*y - tension*(d^2/dx^2)*y = rhs."""
    ds = lspan / (n - 1)

    if rhs is None:
        rhs = np.zeros(n)

    A4 = ei * FD.fourth_derivative(n, ds)
    A2 = -tension * FD.second_derivative(n, ds)
    BC, rhs_bc = bc.compute(ds, n)
    A2 = clean_matrix(bc.order, A2)
    rhs = clean_rhs(bc.order, np.copy(rhs))
    A = A4 + A2 + BC
    rhs_tot = rhs + rhs_bc

    sol = sp.sparse.linalg.spsolve(A, rhs_tot)
    return sol


def compute_curvature(n: int, ds: float, y: np.ndarray[float]) -> np.ndarray[float]:
    """Compute the exact curvature for a given array."""
    y_second = FD.second_derivative(n, ds) @ y
    y_first = FD.first_derivative(n, ds) @ y
    return y_second * (np.ones(n) + y_first**2) ** (-3 / 2.0)


def fixed_point_algo(n,ds,ei,tension,Y0,D2,BC,rhs,rhs_bc):
    sol_old = Y0

    def rhs_fixed_point(y):
        curv = compute_curvature(n, ds, y)
        return rhs + rhs_bc - ei * D2 @ curv
    
    A = - tension * D2 + BC 
    error = 10 
    k = 1
    while k < 3 and error > 1e-4:
        sol_new = sp.sparse.linalg.spsolve(A, rhs_fixed_point(sol_old))
        sol_old = sol_new
        k += 1
    
    return sol_new


def _solve_curvature_exact(
    n: int,
    bc: FD.BoundaryCondition,
    lspan: float = 400.0,
    tension: float = 1.853e04,
    ei: float = 2155.0,
    rhs: Optional[np.ndarray[float]] = None,
) -> Optional[np.ndarray[float]]:
    """Solve equation of the form : ei*(d^2/dx^2)*C(y) - tension*(d^2/dx^2)*y = rhs, where C(y) is the curvature."""

    if rhs is None:
        rhs = np.zeros(n)

    ds = lspan / (n - 1)
    Y0 = _solve_curvature_approx(n, bc, lspan, tension, ei, rhs)

    D2 = FD.second_derivative(n, ds)
    D2 = clean_matrix(bc.order, D2)

    rhs = clean_rhs(bc.order, np.copy(rhs))
    BC, rhs_bc = bc.compute(ds, n)

    def equation(y):
        curv = compute_curvature(n, ds, y)
        return ei * D2 @ curv - tension * D2 @ y + BC @ y - rhs - rhs_bc
    
    # sol = fixed_point_algo(n,ds,ei,tension,Y0,D2,BC,rhs,rhs_bc)
    sol = sp.optimize.root(equation, Y0)

    if not sol.success:
        print(sol.message)

    return sol.x
