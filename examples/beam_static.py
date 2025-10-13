import numpy as np
import matplotlib.pyplot as plt

from slenderpy.future.beam.static import _solve_curvature_approx
import slenderpy.future.beam.fd_utils as FD


def curvature_approx_bending_const():
    # n = total number of points (with extremities)
    n = 100
    ds = 1 / (n - 1)
    x = np.linspace(0, 1, n)

    def exact_solution(x):
        A = -1 / (np.exp(1) ** 2 - 1)
        B = np.exp(1) ** 2 / (np.exp(1) ** 2 - 1)
        D = -B - A
        C = -D - A * np.exp(1) - B * np.exp(-1)
        return A * np.exp(x) + B * np.exp(-x) + C * x + D

    ### solve y"" - y" = 0 with BC ###

    EI = 1
    H = 1

    left = [[1, 0, 0, 0], [0, 0, 1, 1]]
    right = [[1, 0, 0, 0], [0, 0, 1, 0]]
    bc = FD.BoundaryCondition(4, left, right)
    function2 = exact_solution(x)
    sol2 = _solve_curvature_approx(n, ds, EI, H, bc)

    plt.plot(x, function2, color="blue", label="exact")
    plt.plot(x, sol2, color="orange", label="approx")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    curvature_approx_bending_const()
