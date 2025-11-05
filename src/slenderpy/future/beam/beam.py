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


def _solve_dynamic_approx_curvature(
    nb_space: int,
    dt: float,
    initial_time: int,
    final_time: int,
    bc: FD.BoundaryCondition,
    beam: Beam,
    initial_position: np.ndarray[float],
    initial_velocity: np.ndarray[float],
    force: callable,
) -> np.ndarray[float]:
    """Solve equation of the form : m*(d^2/dt^2)*y (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the approximated curvature i.e. (d^2/dx^2)*y.
    """

    lspan = beam.length
    ds = lspan / (nb_space - 1)
    dt2 = dt*0.5
    x = np.linspace(0.,lspan,nb_space)

    y_old = initial_position
    v_old = initial_velocity
    y_all_time = [y_old]

    D2 = FD.second_derivative(nb_space, ds)
    D2 = FD.clean_matrix(bc.order, D2)
    D4 = FD.fourth_derivative(nb_space , ds)
    BC, rhs_bc = bc.compute(ds, nb_space)

    Id = sp.sparse.identity(nb_space)
    Id.data[0,0] = 0
    Id.data[0,1] = 0
    Id.data[0,-1] = 0
    Id.data[0,-2] = 0

    ei = beam.ei_max
    H = beam.tension 
    mass = beam.mass 

    K = ei*D4 - H*D2
    M = mass*Id
    A = M + dt2**2*K  + BC
    B = M - dt2**2*K

    current_time = initial_time

    for _ in range(final_time):
        force_previsous = FD.clean_rhs(bc.order, force(x, current_time))
        force_current = FD.clean_rhs(bc.order, force(x, current_time + dt))

        rhs = B@v_old + dt2*(force_previsous + force_current) - dt*K@y_old  + rhs_bc

        v_new = sp.sparse.linalg.spsolve(A,rhs)
        y_new = y_old + dt2*(v_old + v_new)
        y_new[0:2] = v_new[0:2]
        y_new[-2:] = v_new[-2:]

        current_time += dt

        v_old = v_new
        y_old = y_new

        y_all_time.append(y_new) 
  
        if bc.dynamic_values is not None:
            rhs_bc = bc.update_rhs(nb_space,x,current_time)

    return np.array(y_all_time)