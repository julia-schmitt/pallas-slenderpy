import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from slenderpy.future.beam.dynamic import _solve_curvature_approx, _solve_curvature_exact
import slenderpy.future.beam.static as ST
import slenderpy.future.beam.fd_utils as FD


def test_dynamic_approx_curvature_static_BC():
    nb_space = 200
    dt = 1/10000. 
    final_time = 10000
    mass = 1.
    tension = 1.
    ei_max = 0.
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
        # return -mass*np.cos(t)*x**2*(x-lmax)**2 + ei_max*24.*f(t) - tension*f(t)*(12*x**2 -12*lmax*x + 2*lmax**2)
        return -mass*np.cos(t)*x**2*(x-lmax)**2 - tension*f(t)*(12*x**2 -12*lmax*x + 2*lmax**2)
    

    left = [[1, 0, 0, 1], [0, 1, 0, 0]]
    right = [[1, 0, 0, 1], [0, 1, 0, 0]]
    bc_y = FD.BoundaryCondition(4, left, right)

    left = [[1, 0, 0, 0], [0, 1, 0, 0]]
    right = [[1, 0, 0, 0], [0, 1, 0, 0]]
    bc_v = FD.BoundaryCondition(4, left, right)

    # sol = solve_CN(nb_space = nb_space,dt = dt, lspan = lspan, final_time = final_time, force = force, initial_position = exact, initial_velocity = exact_time_derivative)

    sol = _solve_curvature_approx(nb_space = nb_space, dt = dt, bc_y = bc_y, bc_v = bc_v,
                                lspan = lspan, final_time = final_time, mass = mass, tension = tension,
                                ei_max = ei_max, force = force, initial_position = exact, initial_velocity = exact_time_derivative)
    

    t = 2000
    
    # plt.plot(x,exact(x, t*dt),color = 'blue', label = 'Analytical solution') 
    # plt.plot(x,exact_time_derivative(x,t*dt),color = 'green', label = 'Analytical time derivative solution')
    # plt.plot(x,sol[t], color = 'orange', label = 'Approximate solution')
    # plt.legend()
    # plt.xlim(lmin, lmax)
    # plt.ylim(-7,7)
    # plt.show()

    # fig = plt.figure()
    # line_exact, = plt.plot([], [], color='blue', label = 'Analytical solution')
    # line_approx, = plt.plot([], [], color='orange', label = 'Approximate solution')
    # plt.legend()
    # plt.xlim(lmin, lmax)
    # plt.ylim(-7,7)

    # def animate(i):
    #     t = i*dt
    #     analytical = exact(x,t)
    #     approx = sol[i]
    #     line_exact.set_data(x, analytical)
    #     line_approx.set_data(x,approx)
    #     return line_exact

    # ani = animation.FuncAnimation(fig,animate,frames = np.arange(1,final_time,100), interval=1)
    # ani.save("animation.mp4", writer="ffmpeg", fps=30)

    # plt.show()

    analitical_results = np.array([exact(x,i*dt) for i in range(final_time + 1)])

    atol = 1.0e-06
    rtol = 1.0e-04

    assert np.allclose(analitical_results, sol, atol=atol, rtol=rtol)





def test_dynamic_approx_curvature():
    nb_space = 1000
    nb_time = 100
    final_time = 10.
    dt = final_time/nb_time
    mass = 1.
    tension = 1.
    ei_max = 1.
    lmin = 0. 
    lmax = 8.
    lspan = lmax - lmin 

    x = np.linspace(lmin, lmax, nb_space)

    def exact(x,t):
        return np.cos(2*np.pi*t - x)

    def exact_time_derivative(x,t):
        return -2.*np.pi*np.sin(2*np.pi*t - x)

    def exact_space_derivative(x,t):
        return np.sin(2*np.pi*t - x)
    
    def exact_space_time_derivative(x,t):
        return 2*np.pi*np.cos(2*np.pi*t - x)
    

    def force(x,t):
        return -mass*4.*np.pi**2*exact(x,t) + ei_max*exact(x,t) + tension*exact(x,t) 

    left = [[1, 0, 0, exact_time_derivative(lmin, dt)], [0, 1, 0, exact_time_derivative(lmin, dt)]]
    right = [[1, 0, 0, exact_space_time_derivative(lmax, dt)], [0, 1, 0, exact_space_time_derivative(lmax, dt)]]
    bc_y = FD.BoundaryCondition(4, left, right)

    sol = _solve_curvature_approx(nb_space = nb_space, nb_time = nb_time, bc_y = bc_y, bc_v = bc_y,
                                lspan = lspan, final_time = final_time, tension = tension,
                                ei_max = ei_max, force = force, inital_position = exact, initial_velocity = exact_time_derivative)
    

    plt.plot(x,exact(x,1*dt),color = 'blue', label = 'Analytical solution')
    # plt.plot(x,exact_time_derivative(x,1*dt),color = 'green', label = 'Analytical time derivative solution')
    plt.plot(x,sol[1], color = 'orange', label = 'Approximate solution')
    plt.legend()
    plt.show()

    # fig = plt.figure()
    # line_exact, = plt.plot([], [], color='blue', label = 'Analytical solution')
    # line_approx, = plt.plot([], [], color='orange', label = 'Approximate solution')
    # plt.legend()
    # plt.xlim(lmin, lmax)
    # plt.ylim(-2,2)

    # def animate(i):
    #     t = i*dt
    #     analytical = exact(x,t + dt)
    #     approx = sol[i]
    #     line_exact.set_data(x, analytical)
    #     line_approx.set_data(x,approx)
    #     return line_exact

    # ani = animation.FuncAnimation(fig,animate,frames = 1000,interval=1)

    # plt.show()


def test_dynamic_exact_curvature():
    nb_space = 10000
    nb_time = 10000
    final_time = 10.
    dt = final_time/nb_time
    mass = 1.
    tension = 1.
    ei_max = 1.
    lmin = 0. 
    lmax = 2.
    lspan = lmax - lmin 
    x = np.linspace(lmin, lmax, nb_space)


    def rhs(x,t):
        return np.cosh(x + t) + ei_max*( -2 / np.cosh(x + t)**2 + 6.0 * np.sinh(x + t)**2 / np.cosh(x + t) ** 4) - tension*np.cosh(x + t)

    def exact(x,t):
        return np.cosh(x + t)

    def exact_time_derivative(x,t):
        return np.sinh(x + t)

    left = [[1, 0, 0, np.cosh(lmin + dt)], [0, 1, 0, np.sinh(lmin + dt)]]
    right = [[1, 0, 0, np.cosh(lmax + dt)], [0, 1, 0, np.sinh(lmax + dt)]]
    bc_y = FD.BoundaryCondition(4, left, right)

    sol_curv = _solve_curvature_exact(nb_space = nb_space, nb_time = nb_time, bc = bc_y, 
                                lspan = lspan, final_time = final_time, tension = tension,
                                ei_max = ei_max, force = rhs, inital_position = exact, initial_velocity = exact_time_derivative)


    plt.plot(x,sol_curv[2],color = 'orange', label = 'Approximate solution')
    plt.plot(x,exact(x,3*dt),color = 'blue', label = 'Analytical solution')
    plt.show()

    # fig = plt.figure()
    # line_exact, = plt.plot([], [], color='blue')
    # line_approx, = plt.plot([], [], color='orange')
    # plt.xlim(lmin, lmax)
    # plt.ylim(-2,2)

    # def animate(i):
    #     t = i*dt
    #     analytical = exact(x,t + dt)
    #     approx = sol[i]
    #     line_exact.set_data(x, analytical)
    #     line_approx.set_data(x,approx)
    #     return line_exact

    # ani = animation.FuncAnimation(fig,animate,frames = nb_time,interval=1)

    # plt.show()

    

if __name__ == "__main__":
    # test_scheme_CN()
    test_dynamic_approx_curvature_static_BC()
    # test_dynamic_approx_curvature()
    # test_dynamic_exact_curvature()