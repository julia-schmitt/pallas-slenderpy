from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD


def clean_matrix(n: int, order: int, A2: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
    if order == 4:
        A2.data[0, 0] = 0
        A2.data[0, -3] = 0

        A2.data[1, 1] = 0
        A2.data[1, -2] = 0

        A2.data[2, -1] = 0
        A2.data[2, 2] = 0

    return A2


def clean_rhs(
    n: int, order: int, rhs: Optional[np.ndarray[float]] = None
) -> np.ndarray[float]:
    if rhs is None:
        return np.zeros(n)

    rhs[0] = 0
    rhs[-1] = 0

    if order == 4:
        rhs[1] = 0
        rhs[-2] = 0

    return rhs


def _solve_curvature_approx(
    n: int,
    ds: float,
    EI: float,
    H: float,
    bc: FD.BoundaryCondition,
    rhs: Optional[np.ndarray[float]] = None,
) -> Optional[np.ndarray[float]]:
    ### equation : EI*(d^4/dx^4)*y - H*(d^2/dx^2)*y = F(x) ###
    A4 = EI * FD.fourth_derivative(n, ds)
    A2 = -H * FD.second_derivative(n, ds)  # si H = 0 D2 est quand même calculé ?
    BC, rhs_bc = bc.compute(ds, n)
    A2 = clean_matrix(n, bc.order, A2)
    rhs = clean_rhs(n, ds, rhs)
    A = A4 + A2 + BC
    rhs_tot = rhs + rhs_bc
    sol = sp.sparse.linalg.spsolve(A, rhs_tot)
    return sol
