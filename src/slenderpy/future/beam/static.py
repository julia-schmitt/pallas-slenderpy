from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
from slenderpy.future._constant import _GRAVITY


def clean_matrix(order: int, A2: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
    if order == 4:
        A2.data[0, 0] = 0
        A2.data[0, -3] = 0

        A2.data[1, 1] = 0
        A2.data[1, -2] = 0

        A2.data[2, -1] = 0
        A2.data[2, 2] = 0

    return A2


def clean_rhs(order: int, rhs: Optional[np.ndarray[float]] = None) -> np.ndarray[float]:

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
    tratio: float = 0.17,
    rts: float = 1.853e05,
    EI: float = 2155.0,
    rhs: Optional[np.ndarray[float]] = None,
) -> Optional[np.ndarray[float]]:

    ### equation : EI*(d^4/dx^4)*y - H*(d^2/dx^2)*y = F(x) ###
    ds = lspan / (n - 1)
    H = rts * tratio

    if rhs is None:
        rhs = _GRAVITY * np.ones(n)

    A4 = EI * FD.fourth_derivative(n, ds)
    A2 = -H * FD.second_derivative(n, ds)
    BC, rhs_bc = bc.compute(ds, n)
    A2 = clean_matrix(bc.order, A2)
    rhs = clean_rhs(bc.order, np.copy(rhs))
    A = A4 + A2 + BC
    rhs_tot = rhs + rhs_bc

    sol = sp.sparse.linalg.spsolve(A, rhs_tot)
    return sol


def compute_curvature(n: int, ds: float, y: np.ndarray[float]) -> np.ndarray[float]:
    y_second = FD.second_derivative(n, ds) @ y
    y_first = FD.first_derivative(n, ds) @ y
    return y_second * np.power((np.ones(n) + np.power(y_first, 2)), -3 / 2.0)


def _solve_curvature_exact(
    n: int,
    bc: FD.BoundaryCondition,
    lspan: float = 400.0,
    tratio: float = 0.17,
    rts: float = 1.853e05,
    EI: float = 2155.0,
    rhs: Optional[np.ndarray[float]] = None,
) -> Optional[np.ndarray[float]]:

    ds = lspan / (n - 1)
    H = rts * tratio
    Y0 = _solve_curvature_approx(n, bc, lspan, tratio, rts, EI, rhs)

    D2 = FD.second_derivative(n, ds)
    D2 = clean_matrix(bc.order, D2)

    rhs = clean_rhs(bc.order, np.copy(rhs))
    BC, rhs_bc = bc.compute(ds, n)

    def equation(y):
        curv = compute_curvature(n, ds, y)
        return EI * D2 @ curv - H * D2 @ y + BC @ y - rhs - rhs_bc

    sol = sp.optimize.root(equation, Y0)
    # hybr lm krylov

    return sol.x
