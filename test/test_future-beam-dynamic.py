import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import slenderpy.future.beam.beam as Beam
from slenderpy.future.beam.fd_utils import BoundaryCondition


def _plot_animcation(x, exact, sol, ymin, ymax, dt, final_time):
    """Animation to plot the analytical and the numerical solution."""

    fig = plt.figure()
    line_exact, = plt.plot([], [], color='blue', label = 'Analytical solution')
    line_approx, = plt.plot([], [], color='orange', label = 'Approximate solution')
    plt.legend()
    plt.xlim(x[0], x[-1])
    plt.ylim(-7,7)

    def animate(i):
        t = i*dt
        analytical = exact(x,t)
        approx = sol[i]
        line_exact.set_data(x, analytical)
        line_approx.set_data(x,approx)
        return line_exact

    ani = animation.FuncAnimation(fig,animate,frames = np.arange(1,final_time,100), interval=1)

    plt.show()


def test_solve_cruvature_approx_static_BC(plot=False):
    nb_space = 1000
    dt = 1/10000. 
    final_time = 10000
    mass = 14.3
    tension = 125.78
    ei_max = 1484.75
    lmin = 0. 
    lmax = 3.
    lspan = lmax - lmin 

    x = np.linspace(lmin, lmax, nb_space)

    def f(t):
        return np.cos(t)

    def exact(x,t):
        return f(t)*x**2*(x-lmax)**2 + 1
    
    def exact_time_derivative(x,t):
        return -np.sin(t)*x**2*(x-lmax)**2
    
    def force(x,t):
        return -mass*np.cos(t)*x**2*(x-lmax)**2 + ei_max*24.*f(t) - tension*f(t)*(12*x**2 -12*lmax*x + 2*lmax**2)    

    left = [[1, 0, 0, 1], [0, 1, 0, 0]]
    right = [[1, 0, 0, 1], [0, 1, 0, 0]]
    bc = BoundaryCondition(4, left, right)

    beam = Beam.Beam(length = lspan, tension = tension, ei_max = ei_max, mass = mass)


    sol = Beam._solve_dynamic_approx_curvature(nb_space = nb_space, dt = dt, initial_time = 0, final_time = final_time,
                                          bc = bc, beam = beam, initial_position = exact(x,0), initial_velocity = exact_time_derivative(x,0),
                                          force = force)
    
    if plot:
        _plot_animcation(x, exact, sol, -7, 7, dt, final_time)

    analitical_results = np.array([exact(x,i*dt) for i in range(final_time + 1)])
    atol = 1.0e-02
    rtol = 1.0e-02

    assert np.allclose(analitical_results, sol, atol=atol, rtol=rtol)


def test_solve_cruvature_approx_dynamic_BC(plot=False):
    nb_space = 1000
    dt = 1/10000. 
    final_time = 10000
    mass = 1.45
    tension = 12.36
    ei_max = 147.89
    lmin = 0. 
    lmax = 4.
    lspan = lmax - lmin 

    x = np.linspace(lmin, lmax, nb_space)

    def exact(x,t):
        return np.cosh(x - 2)*np.sin(2*np.pi*t) 
    
    def exact_space_derivative(x,t):
        return np.sinh(x - 2)*np.sin(2*np.pi*t)
    
    def exact_time_derivative(x,t):
        return 2*np.pi*np.cosh(x - 2)*np.cos(2*np.pi*t)
    
    def force(x,t):
        return -4*np.pi**2*mass*exact(x,t) + ei_max*exact(x,t) - tension*exact(x,t)

    initial_time = 0 + dt

    left = [[1, 0, 0, exact(lmin,initial_time)], [0, 1, 0, exact_space_derivative(lmin,initial_time)]]
    right = [[1, 0, 0, exact(lmax,initial_time)], [0, 1, 0, exact_space_derivative(lmax,initial_time)]]
    dynamic_values = [exact,exact_space_derivative,exact_space_derivative,exact]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    beam = Beam.Beam(length = lspan, tension = tension, ei_max = ei_max, mass = mass)
    sol = Beam._solve_dynamic_approx_curvature(nb_space = nb_space, dt = dt, initial_time = initial_time, final_time = final_time,
                                          bc = bc, beam = beam, initial_position = exact(x,0), initial_velocity = exact_time_derivative(x,0),
                                          force = force)

    if plot:
        _plot_animcation(x, exact, sol, -1, 5, dt, final_time)

    analitical_results = np.array([exact(x,i*dt) for i in range(final_time + 1)])
    atol = 1.0e-02
    rtol = 1.0e-02

    assert np.allclose(analitical_results, sol, atol=atol, rtol=rtol)