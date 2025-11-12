import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import slenderpy.future.beam.beam as Beam
from slenderpy.future.beam.fd_utils import BoundaryCondition


def _plot_animation(x, exact, sol, ymin, ymax, nb_time, initial_time, final_time):
    """Animation to plot the analytical and the numerical solution."""

    fig = plt.figure()
    line_exact, = plt.plot([], [], color='blue', label = 'Analytical solution')
    line_approx, = plt.plot([], [], color='orange', label = 'Approximate solution')
    plt.legend()
    plt.xlim(x[0], x[-1])
    plt.ylim(ymin,ymax)

    dt = final_time / nb_time
    print(dt)

    def animate(i):
        t = initial_time + i*dt
        analytical = exact(x,t)
        approx = sol[i]
        line_exact.set_data(x, analytical)
        line_approx.set_data(x,approx)
        return line_exact, line_approx,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,nb_time + 1,10),interval=1, blit=True, repeat=True)
    # ani.save("exact_curvature_picard.mp4", writer="ffmpeg", fps=100)
    plt.show()


def test_solve_approx_curvature_static_BC(plot=False):
    nb_space = 800
    nb_time = 20000
    final_time = 5
    mass = 14.3
    tension = 125.78
    ei_max = 1484.75
    lmin = 0. 
    lmax = 3.
    lspan = lmax - lmin 

    x = np.linspace(lmin, lmax, nb_space)
    initial_time = 0

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


    sol = Beam._solve_dynamic_approx_curvature(nb_space = nb_space, nb_time = nb_time, initial_time = initial_time, final_time = final_time,
                                          bc = bc, beam = beam, initial_position = exact(x,0), initial_velocity = exact_time_derivative(x,0),
                                          force = force)
    
    if plot:
        _plot_animation(x, exact, sol, -7, 7, nb_time, initial_time, final_time)

    analitical_results = np.array([exact(x,initial_time + i*(final_time/nb_time)) for i in range(nb_time + 1)])
    print(np.shape(analitical_results))
    print(np.shape(sol))
    atol = 1.0e-01
    rtol = 1.0e-01

    assert np.allclose(analitical_results, sol, atol=atol, rtol=rtol)


def test_solve_approx_curvature_dynamic_BC(plot=False):
    nb_space = 1000
    nb_time = 20000
    final_time = 2
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

    initial_time = 0.

    left = [[1, 0, 0, exact(lmin,initial_time)], [0, 1, 0, exact_space_derivative(lmin,initial_time)]]
    right = [[1, 0, 0, exact(lmax,initial_time)], [0, 1, 0, exact_space_derivative(lmax,initial_time)]]
    dynamic_values = [exact,exact_space_derivative,exact_space_derivative,exact]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    beam = Beam.Beam(length = lspan, tension = tension, ei_max = ei_max, mass = mass)
    sol = Beam._solve_dynamic_approx_curvature(nb_space = nb_space, nb_time = nb_time, initial_time = initial_time, final_time = final_time,
                                          bc = bc, beam = beam, initial_position = exact(x,0), initial_velocity = exact_time_derivative(x,0),
                                          force = force)

    if plot:
        _plot_animation(x, exact, sol, -5, 5, nb_time, initial_time, final_time)

    analitical_results = np.array([exact(x,initial_time + i*(final_time/nb_time)) for i in range(nb_time + 1)])
    atol = 1.0e-02
    rtol = 1.0e-02

    assert np.allclose(sol, analitical_results,atol=atol, rtol=rtol)


def test_solve_exact_curvature(plot=False):
    nb_space = 100
    nb_time = 10000
    final_time = 1
    dt = final_time/nb_time
    mass = 1.
    tension = 1.
    ei_max = 1.
    lmin = 0. 
    lmax = 2.
    lspan = lmax - lmin 
    x = np.linspace(lmin, lmax, nb_space)

    def force(x,t):
        return np.cosh(x + t) + ei_max*( -2 / np.cosh(x + t)**2 + 6.0 * np.sinh(x + t)**2 / np.cosh(x + t) ** 4) - tension*np.cosh(x + t)

    def exact(x,t):
        return np.cosh(x + t)
    
    def exact_space_derivative(x,t):
        return np.sinh(x + t)

    def exact_time_derivative(x,t):
        return np.sinh(x + t)
    
    initial_time = 0.0

    left = [[1, 0, 0, exact(lmin,initial_time)], [0, 1, 0, exact_space_derivative(lmin,initial_time)]]
    right = [[1, 0, 0, exact(lmax,initial_time)], [0, 1, 0, exact_space_derivative(lmax,initial_time)]]
    dynamic_values = [exact,exact_space_derivative,exact_space_derivative,exact]
    bc = BoundaryCondition(4, left, right, dynamic_values)

    beam = Beam.Beam(length = lspan, tension = tension, ei_max = ei_max, mass = mass)
    sol = Beam._solve_dynamic_exact_curvature(nb_space = nb_space, nb_time = nb_time, initial_time = initial_time, final_time = final_time,
                                          bc = bc, beam = beam, initial_position = exact(x,initial_time), initial_velocity = exact_time_derivative(x,initial_time),
                                          force = force)

    if plot:
        _plot_animation(x, exact, sol, 1, 7, nb_time, initial_time, final_time)

    analitical_results = np.array([exact(x,initial_time + i*(final_time/nb_time)) for i in range(nb_time + 1)])
    atol = 1.0e-06
    rtol = 1.0e-02

    assert np.allclose(sol, analitical_results,atol=atol, rtol=rtol)



if __name__ == '__main__':
    test_solve_exact_curvature()

