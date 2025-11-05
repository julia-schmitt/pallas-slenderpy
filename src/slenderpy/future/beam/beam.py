from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD


class Beam:
    """A Beam object."""

    def __init__(
        self,
        length: float,
        tension: float,
        ei_max: float,
        ei_min: Optional[float] = None,
        critical_curvature: Optional[float] = None,
        mass: Optional[float] = None,
    ) -> None:

        self.length = length
        self.tension = tension
        self.ei_max = ei_max
        self.ei_min = ei_min
        self.critical_curvature = critical_curvature
        self.mass = mass


def _solve_static_approx_curvature(
    n: int,
    bc: FD.BoundaryCondition,
    beam: Beam,
    rhs: np.ndarray[float],
) -> np.ndarray[float]:
    """Solve equation of the form : (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the approximated curvature i.e. (d^2/dx^2)*y.
    """
    ds = beam.length / (n - 1)

    ei = beam.ei_max
    H = beam.tension

    D2_order = FD.second_derivative(n, ds)
    D2 = FD.clean_matrix(bc.order, D2_order)
    BC, rhs_bc = bc.compute(ds, n)
    D4 = FD.fourth_derivative(n, ds)
    K = ei * D4 - H * D2
    rhs = FD.clean_rhs(bc.order, rhs)

    A = K + BC
    rhs_tot = rhs + rhs_bc

    sol = sp.sparse.linalg.spsolve(A, rhs_tot)

    if beam.critical_curvature is not None:

        def equation(y):
            curvature = D2_order @ y
            bending_moment = compute_bending_moment(
                curvature, beam.ei_max, beam.ei_min, beam.critical_curvature
            )
            return D2 @ bending_moment - beam.tension * D2 @ y + BC @ y - rhs_tot

        result = sp.optimize.root(equation, sol)

        if not result.success:
            print(result.message)

        sol = result.x

    return sol


def compute_curvature(n: int, ds: float, y: np.ndarray[float]) -> np.ndarray[float]:
    """Compute the exact curvature for a given array."""
    y_second = FD.second_derivative(n, ds) @ y
    y_first = FD.first_derivative(n, ds) @ y
    return y_second * (np.ones(n) + y_first**2) ** (-3 / 2.0)


def compute_bending_moment(
    curvature: np.ndarray[float],
    ei_max: float,
    ei_min: float,
    critical_curvature: float,
) -> np.ndarray[float]:
    """Compute the bending moment if not constant, otherwise return ei_max."""
    if critical_curvature is None:
        return ei_max * curvature

    else:
        chi_bar = (1 - ei_min / ei_max) * critical_curvature
        return (ei_max * chi_bar + ei_min * curvature) * (
            1 - np.exp(-curvature / chi_bar)
        )


def _solve_static_exact_curvature(
    n: int, bc: FD.BoundaryCondition, beam: Beam, rhs: np.ndarray[float]
) -> np.ndarray[float]:
    """Solve equation of the form : (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the exact curvature.
    """

    ds = beam.length / (n - 1)
    Y0 = _solve_static_approx_curvature(n, bc, beam, rhs)

    D2 = FD.second_derivative(n, ds)
    D2 = FD.clean_matrix(bc.order, D2)

    rhs = FD.clean_rhs(bc.order, rhs)
    BC, rhs_bc = bc.compute(ds, n)

    def equation(y):
        curvature = compute_curvature(n, ds, y)
        bending_moment = compute_bending_moment(
            curvature, beam.ei_max, beam.ei_min, beam.critical_curvature
        )

        return D2 @ bending_moment - beam.tension * D2 @ y + BC @ y - rhs - rhs_bc

    sol = sp.optimize.root(equation, Y0)

    if not sol.success:
        print(sol.message)

    return sol.x
