from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
import slenderpy.future.beam.static as ST

def _solve_curvature_approx(
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
    inital_position = None,
    initial_velocity = None
) -> sp.sparse.spmatrix:
    """Solve equation of the form : m*(d^2/dt^2)*y + ei*(d^4/dx^4)*y - tension*(d^2/dx^2)*y = rhs."""

    ds = lspan / (nb_space - 1)
    dt = final_time / nb_time
    dt2 = dt/2.
    x = np.linspace(0.,lspan,nb_space)

    t = 0.0
    y_old = inital_position
    y_prime_old = initial_velocity

    y_all_time = []

    ei = ei_max

    D2 = FD.second_derivative(nb_space, ds)
    D2 = FD.clean_matrix(bc.order, D2)
    D4 = FD.fourth_derivative(nb_space, ds)
    BC, rhs_bc = bc.compute(ds, nb_space)

    for _ in range(100):
        force_previsous = FD.clean_rhs(bc.order, np.copy(force(x, t)))
        force_current = FD.clean_rhs(bc.order, np.copy(force(x, t + dt)))
        y_prime_old = FD.clean_rhs(bc.order, np.copy(y_prime_old))

        equation = ei*D4 - tension*D2

        rhs = mass*y_prime_old + dt2*(force_previsous + force_current - 2*equation@y_old)  - dt2**2*equation@y_prime_old + rhs_bc
        Id = sp.sparse.identity(nb_space)
        Id.data[0,0] = 0
        Id.data[0,1] = 0
        Id.data[0,-1] = 0
        Id.data[0,-2] = 0
        A =  mass*Id + dt2**2*equation + BC

        y_prime_new = sp.sparse.linalg.spsolve(A,rhs) 
        y_new = y_old + dt2*(y_prime_old + y_prime_new)
        y_new[0:2] = y_new[0:2]  
        y_new[-2::] = y_new[-2::]

        y_prime_old = y_prime_new
        y_old = y_new
        t += dt

        # rhs_bc[0] = -2.*np.pi*np.sin(2*np.pi*t)
        # rhs_bc[1] = 2*np.pi*np.cos(2*np.pi*t)
        # rhs_bc[-1] = -2.*np.pi*np.sin(2*np.pi*t - lspan)
        # rhs_bc[-2] = 2*np.pi*np.cos(2*np.pi*t - lspan)

        y_all_time.append(y_prime_new)


    return y_all_time

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
    inital_position = None,
    initial_velocity = None
) -> sp.sparse.spmatrix:
    """Solve equation of the form : m*(d^2/dt^2)*y + ei*(d^2/dx^2)*C(y) - tension*(d^2/dx^2)*y = rhs, where C(y) is the curvature."""

    ds = lspan / (nb_space - 1)
    dt = final_time / nb_time
    x = np.linspace(0.,lspan,nb_space)
    dt2 = dt/2.

    t = 0.0
    y_old = inital_position(x,t)
    y_prime_old = initial_velocity(x,t)

    y_all_time = []

    D2 = FD.second_derivative(nb_space, ds)
    D2 = FD.clean_matrix(bc.order, D2)

    BC, rhs_bc = bc.compute(ds, nb_space)

    for _ in range(10):
        t += dt
        
        curvature = ST.compute_curvature(nb_space, ds, y_old)
        bending_moment = ST.compute_bending_moment(curvature, ei_max, ei_min, critical_curvature)

        force_previsous = FD.clean_rhs(bc.order, np.copy(force(y_old, t)))
        force_current = FD.clean_rhs(bc.order, np.copy(force(y_old, t + dt)))
        y_prime_old = FD.clean_rhs(bc.order, np.copy(y_prime_old))

        rhs = y_prime_old + dt2*(force_previsous + force_current - 2*D2@bending_moment + 2*tension*D2 @ y_old)  + dt2**2*tension*D2@y_prime_old + rhs_bc
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
         
        rhs_bc[0] = np.cosh(t)
        rhs_bc[1] = np.sinh(t)
        rhs_bc[-1] = np.cosh(lspan + t)
        rhs_bc[-2] = np.sinh(lspan + t)

        y_all_time.append(y_new)

    return y_all_time