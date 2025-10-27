from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
import slenderpy.future.beam.static as ST

def _solve_curvature_exact(
    nb_space: int,
    nb_time: int,
    bc: FD.BoundaryCondition,
    lspan: float = 400.0,
    final_time : float = 30.0,
    mass : float = 10., #tout diviser par la masse 
    tension: float = 1.853e04,
    ei_max: float = 2155.0,
    ei_min: float = None,
    critical_curvature = None, 
    force = None, #a modifier 
) -> sp.sparse.spmatrix:
    """Solve equation of the form : m*(d^2/dt^2)*y + ei*(d^2/dx^2)*C(y) - tension*(d^2/dx^2)*y = rhs, where C(y) is the curvature."""

    ds = lspan / (nb_space - 1)
    dt = final_time / nb_time
    print("ds**2 = ", ds**2)
    print("dt = ", dt)
    dt2 = dt/2.

    t = 0.0
    y_old = np.linspace(0,lspan,nb_space)
    y_prime_old = np.zeros(nb_space)

    y_all_time = []

    D2 = FD.second_derivative(nb_space, ds)
    D2 = FD.clean_matrix(bc.order, D2)

    BC, rhs_bc = bc.compute(ds, nb_space)

    for _ in range(2):
        curvature = ST.compute_curvature(nb_space, ds, y_old)
        bending_moment = ST.compute_bending_moment(curvature, ei_max, ei_min, critical_curvature)

        force_previsous = FD.clean_rhs(bc.order, np.copy(force(y_old, t)))
        force_current = FD.clean_rhs(bc.order, np.copy(force(y_old, t + dt)))

        rhs = y_prime_old + dt2*(force_previsous + force_current - 2*D2@bending_moment + 2*tension*D2 @ y_prime_old)  + dt**2*tension*D2@y_old + rhs_bc
        Id = sp.sparse.identity(nb_space)
        Id.data[0,0] = 0
        Id.data[0,1] = 0
        Id.data[0,-1] = 0
        Id.data[0,-2] = 0
        A =  Id - dt2**2*tension*D2 + BC

        y_prime_new = sp.sparse.linalg.spsolve(A,rhs) 
        y_new = y_old + dt2*(y_prime_old + y_prime_new)
        y_new[0:2] = y_prime_new[0:2]
        y_new[-2::] = y_prime_new[-2::]

        y_prime_old = y_prime_new
        y_old = y_new
        t += dt 

        y_all_time.append(y_new)

    return y_all_time