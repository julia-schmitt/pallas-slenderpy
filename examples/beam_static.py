import numpy as np
import matplotlib.pyplot as plt

from slenderpy.future.beam.static import _solve_curvature_approx, _solve_curvature_exact
import slenderpy.future.beam.fd_utils as FD


def curvature_approx_bending_const():
    # n = total number of points (with extremities)
    n = 1000
    x = np.linspace(0, 1, n)

    def exact_solution(x):
        A = -1 / (np.exp(1) ** 2 - 1)
        B = np.exp(1) ** 2 / (np.exp(1) ** 2 - 1)
        D = -B - A
        C = -D - A * np.exp(1) - B * np.exp(-1)
        return A * np.exp(x) + B * np.exp(-x) + C * x + D

    ### solve y"" - y" = 0 with BC ###

    left = [[1, 0, 0, 0], [0, 0, 1, 1]]
    right = [[1, 0, 0, 0], [0, 0, 1, 0]]
    bc = FD.BoundaryCondition(4, left, right)
    function = exact_solution(x)
    sol = _solve_curvature_approx(
        n=n, bc=bc, lspan=1, tratio=1, rts=1, EI=1, rhs=np.zeros(n)
    )

    plt.plot(x, function, "--", color="blue", label="analytical")
    plt.plot(x, sol, color="orange", label="FD solution")
    plt.legend()
    plt.show()


def curvature_exact_bending_const():
    n = 1000
    lmin = -1.0
    lmax = 2.0
    lspan = lmax - lmin
    x = np.linspace(lmin, lmax, n)

    def exact_solution(x):
        return np.cosh(x)

    def rhs(x):
        return 1.0 / np.cosh(x) - 2.0 / np.cosh(x) ** 3 - np.cosh(x)

    left = [[1, 0, 0, np.cosh(lmin)], [0, 1, 0, np.sinh(lmin)]]
    right = [[1, 0, 0, np.cosh(lmax)], [0, 1, 0, np.sinh(lmax)]]
    bc = FD.BoundaryCondition(4, left, right)
    sol = _solve_curvature_exact(
        n=n, bc=bc, lspan=lspan, tratio=1, rts=1, EI=1, rhs=rhs(x)
    )
    approx_curvature = _solve_curvature_approx(
        n=n, bc=bc, lspan=lspan, tratio=1, rts=1, EI=1, rhs=rhs(x)
    )
    function = exact_solution(x)

    plt.plot(x, function, "--", color="blue", label="analytical")
    plt.plot(x, sol, color="orange", label="FD solution")
    plt.plot(x, approx_curvature, color="green", label="FD solution approx curvature")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    curvature_approx_bending_const()
    curvature_exact_bending_const()
