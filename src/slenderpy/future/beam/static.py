from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD


def _solve_curvature_approx(
    n: int,
    bc: FD.BoundaryCondition,
    lspan: float = 400.0,
    tension: float = 1.853e04,
    ei_min: float = 2155.0,
    ei_max: float = 2155.0,
    rhs: Optional[np.ndarray[float]] = None,
) -> Optional[np.ndarray[float]]:
    """Solve equation of the form : ei*(d^4/dx^4)*y - tension*(d^2/dx^2)*y = rhs."""
    ds = lspan / (n - 1)

    if rhs is None:
        rhs = np.zeros(n)

    ei = (ei_max + ei_min) / 2

    A4 = ei * FD.fourth_derivative(n, ds)
    A2 = -tension * FD.second_derivative(n, ds)
    BC, rhs_bc = bc.compute(ds, n)
    A2 = FD.clean_matrix(bc.order, A2)
    rhs = FD.clean_rhs(bc.order, np.copy(rhs))
    A = A4 + A2 + BC
    rhs_tot = rhs + rhs_bc

    sol = sp.sparse.linalg.spsolve(A, rhs_tot)
    return sol


def compute_curvature(n: int, ds: float, y: np.ndarray[float]) -> np.ndarray[float]:
    """Compute the exact curvature for a given array."""
    y_second = FD.second_derivative(n, ds) @ y
    y_first = FD.first_derivative(n, ds) @ y
    return y_second * (np.ones(n) + y_first**2) ** (-3 / 2.0)


def compute_bending_moment(curvature : np.ndarray[float], ei_min : float, ei_max : float) -> np.ndarray[float]:
    return (ei_min + ei_max)/2*curvature 


def _solve_curvature_exact(
    n: int,
    bc: FD.BoundaryCondition,
    lspan: float = 400.0,
    tension: float = 1.853e04,
    ei_min: float = 2155.0,
    ei_max: float = 2155.0,
    rhs: Optional[np.ndarray[float]] = None,
) -> Optional[np.ndarray[float]]:
    """Solve equation of the form : ei*(d^2/dx^2)*C(y) - tension*(d^2/dx^2)*y = rhs, where C(y) is the curvature."""

    if rhs is None:
        rhs = np.zeros(n)

    ds = lspan / (n - 1)
    Y0 = _solve_curvature_approx(n, bc, lspan, tension, ei_min, ei_max, rhs)

    D2 = FD.second_derivative(n, ds)
    D2 = FD.clean_matrix(bc.order, D2)

    rhs = FD.clean_rhs(bc.order, np.copy(rhs))
    BC, rhs_bc = bc.compute(ds, n)

    def equation(y):
        curvature = compute_curvature(n, ds, y)
        bending_moment = compute_bending_moment(curvature, ei_min, ei_max)

        return D2 @ bending_moment  - tension * D2 @ y + BC @ y - rhs - rhs_bc
    
    sol = sp.optimize.root(equation, Y0)

    if not sol.success:
        print(sol.message)

    return sol.x
