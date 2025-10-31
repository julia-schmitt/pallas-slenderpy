from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
import slenderpy.future.beam.static as ST


def _solve_curvature_approx(
    nb_space: int,
    dt: float,
    bc_y: FD.BoundaryCondition,
    bc_v: FD.BoundaryCondition,
    lspan: float = 400.0,
    final_time : float = 30.0,
    mass : float = 10., #tout diviser par la masse 
    tension: float = 1.853e04,
    ei_max: float = 2155.0,
    ei_min: float = None,
    critical_curvature = None, 
    force = None, #a modifier 
    initial_position = None,
    initial_velocity = None
) -> sp.sparse.spmatrix:
#ajouter l'option de commencer à n'importe quel temps initial 
    """Solve equation of the form : m*(d^2/dt^2)*y + ei*(d^4/dx^4)*y - tension*(d^2/dx^2)*y = rhs."""

    ds = lspan / (nb_space - 1)
    dt2 = dt*0.5
    x = np.linspace(0.,lspan,nb_space)

    # print("rapport = ", dt/ds**2)

    t = 0.
    y_old = initial_position(x,t)
    v_old = initial_velocity(x,t)
    y_all_time = [y_old]

    ei = ei_max

    D2 = FD.second_derivative(nb_space, ds)
    # D2 = FD.clean_matrix(bc_v.order, D2)
    D4 = FD.fourth_derivative(nb_space , ds)
    # BC, rhs_bc = bc_v.compute(ds, nb_space)
    # BC, rhs_bc = bc_y.compute(ds, nb_space)

    Id = sp.sparse.identity(nb_space)
    # Id.data[0,0] = 0
    # Id.data[0,1] = 0
    # Id.data[0,-1] = 0
    # Id.data[0,-2] = 0

    K = ei*D4 - tension*D2
    M = mass*Id

    for _ in range(final_time):
        # force_previsous = FD.clean_rhs(bc_v.order, np.copy(force(x, t)))
        # force_current = FD.clean_rhs(bc_v.order, np.copy(force(x, t + dt)))
        # v_old = FD.clean_rhs(bc_v.order, np.copy(v_old))
        # y_old = FD.clean_rhs(bc_y.order, np.copy(y_old))

        rhs = (M - dt2**2*K)@v_old + dt2*(force(x,t) + force(x,t + dt)) - dt*K@y_old  #+ rhs_bc
        A =  M + dt2**2*K  #+ BC

        v_new = sp.sparse.linalg.spsolve(A,rhs)
        v_new[0] = 0
        v_new[-1] = 0
        # v_new = v_old + dt2*(force_previsous + force_current - K@y_old - K@y_new)/mass

        # v_new[0:2] = np.array([0,0])
        # v_new[-2::] = np.array([0,0])

        y_new = y_old + dt2*(v_old + v_new)
        y_new[0] = 1
        y_new[-1] = 1
        # y_new[0:2] = np.array([1,1])
        # y_new[-2::] = np.array([1,1])

        v_old = v_new
        y_old = y_new
        t += dt

        y_all_time.append(y_new)

    return np.array(y_all_time)



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