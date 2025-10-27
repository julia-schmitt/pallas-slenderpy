import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from slenderpy.future.beam.dynamic import _solve_curvature_exact
import slenderpy.future.beam.fd_utils as FD
import slenderpy.beam as bm 
import slenderpy.simtools as sm 
import slenderpy.fdm_utils as fdm


nb_space = 100
nb_time = 1
final_time = 1.94
dt = final_time/nb_time
# mass = 1.
tension = 1.
ei_max = 1.
lmin = 0.
lmax = 1.
lspan = lmax - lmin 

x = np.linspace(lmin, lmax, nb_space)

def rhs(x,t):
    return np.cosh(x + t) + ei_max*( -2 / np.cosh(x + t)**2 + 6.0 * np.sinh(x + t)**2 / np.cosh(x + t) ** 4) - tension*np.cosh(x + t)

def exact(x,t):
    return np.cosh(x + t)

def exact_time_derivative(x,t):
    return np.sinh(x + t)


#checker les conditions au bords avec le temps 
t = 0.

left = [[1, 0, 0, np.cosh(lmin + t)], [0, 1, 0, np.sinh(lmin + t)]]
right = [[1, 0, 0, np.cosh(lmax + t)], [0, 1, 0, np.sinh(lmax + t)]]
bc_y = FD.BoundaryCondition(4, left, right)

sol = _solve_curvature_exact(nb_space = nb_space, nb_time = nb_time, bc = bc_y, 
                            lspan = lspan, final_time = final_time, tension = tension,
                            ei_max = ei_max, force = rhs)



# fig = plt.figure()
# line_exact, = plt.plot([], [], color='blue')
# line_approx, = plt.plot([], [], color='orange')
# plt.xlim(lmin, lmax)
# plt.ylim(0,10)

# def animate(i):
#     t = i*dt
#     exact = exact_time_derivative(x,t)
#     approx = sol[i]
#     line_exact.set_data(x, exact)
#     line_approx.set_data(x,approx)
#     return line_exact, line_approx,

# ani = animation.FuncAnimation(fig,animate,frames = nb_time,interval=1)

# plt.show()

plt.plot(x,sol[0], color = 'orange')
plt.plot(x,exact(x,t), color = 'blue')
# # plt.plot(x,exact_time_derivative(x,t), color = 'green')
plt.show()

