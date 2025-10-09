from typing import Tuple

import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt


def second_derivative(n: int, ds: float) -> np.ndarray[[float], [float]]:
    # centered scheme, the first and last line have to be completed with BC 

    dinf = +1.0 * np.ones((n - 1,))/ds**2
    diag = -2.0 * np.ones((n ,))/ds**2
    dsup = +1.0 * np.ones((n - 1,))/ds**2

    dinf[-1] = 0
    diag[0] = 0; diag[-1] = 0
    dsup[0] = 0

    res = sp.sparse.diags([dinf, diag, dsup], [-1, 0, 1])

    return res


def fourth_derivative(n: int, ds: float) -> np.ndarray[[float], [float]]:
    # centered scheme, the two first and two last line have to be completed with BC 

    dinf2 = +1.0 * np.ones((n - 2))/ds**4
    dinf1 = -4.0 * np.ones((n - 1,))/ds**4
    diag = +6.0 * np.ones((n ,))/ds**4
    dsup1 = -4.0 * np.ones((n - 1,))/ds**4
    dsup2 = +1.0 * np.ones((n - 2,))/ds**4

    dinf2[-1] = 0; dinf2[-2] = 0
    dinf1[-1] = 0; dinf1[-2] = 0; dinf1[0] = 0
    diag[0] = 0; diag[1] = 0; diag[-2] = 0; diag[-1] = 0
    dsup1[0] = 0; dsup1[1] = 0; dsup1[-1] = 0
    dsup2[0] = 0; dsup2[1] = 0

    res = sp.sparse.diags([dinf2, dinf1, diag, dsup1, dsup2], [-2, -1, 0, 1, 2])

    return res


def boundary_condition(ds: float, n:int,
    t1: Tuple[float, float, float, float],
    t2: Tuple[float, float, float, float],
    t3: Tuple[float, float, float, float],
    t4: Tuple[float, float, float, float],
    others: bool 
) -> Tuple[np.ndarray[[float], [float]], np.ndarray[float]]:

    a1, b1, c1, d1 = t1
    a2, b2, c2, d2 = t2
    a3, b3, c3, d3 = t3
    a4, b4, c4, d4 = t4

    rankL = np.linalg.matrix_rank(np.array([t1, t2]))
    if rankL < 2:
        raise ValueError("The left boundary conditions are not linearly independant")
    
    rankR = np.linalg.matrix_rank(np.array([t3, t4]))
    if rankR < 2:
        raise ValueError("The right boundary conditions are not linearly independant")

    bc_matrix = np.zeros((n,n))

    bc_matrix[0, 0] = a1 - b1 / ds + c1 / ds**2
    bc_matrix[0, 1] = b1 / ds - 2 * c1 / ds**2
    bc_matrix[0, 2] = c1 / ds**2
    bc_matrix[1, 0] = a2 - b2 / ds + c2 / ds**2 
    bc_matrix[1, 1] = b2 / ds - 2 * c2 / ds**2 
    bc_matrix[1, 2] = c2 / ds**2

    bc_matrix[-1, -1] = a4 + b4 / ds + c4 / ds**2
    bc_matrix[-1, -2] = -b4 / ds - 2 * c4 / ds**2
    bc_matrix[-1, -3] = c4 / ds**2
    bc_matrix[-2, -1] = a3 + b3 / ds + c3 / ds**2 
    bc_matrix[-2, -2] = -b3 / ds - 2 * c3 / ds**2 
    bc_matrix[-2, -3] = c3 / ds**2 

    if others == True:
        bc_matrix[1, 0] -= -1/ds**2
        bc_matrix[1, 1] -= 2/ds**2
        bc_matrix[1, 2] -= -1/ds**2
        bc_matrix[-2, -1] -= -1/ds**2
        bc_matrix[-2, -2] -= 2/ds**2
        bc_matrix[-2, -3] -= -1/ds**2

    bc_matrix = sp.sparse.bsr_matrix(bc_matrix)

    rhs = np.zeros(n)
    rhs[0] = d1
    rhs[1] = d2
    rhs[-1] = d4
    rhs[-2] = d3

    return bc_matrix, rhs


def exact_solutionE1(x):
    return x**3 - x**2 + x


A = -1 / (np.exp(1) ** 2 - 1)
B = np.exp(1) ** 2 / (np.exp(1) ** 2 - 1)
D = -B - A
C = -D - A * np.exp(1) - B * np.exp(-1)


def exact_solutionE2(x):
    return A * np.exp(x) + B * np.exp(-x) + C * x + D


if __name__ == "__main__":
    # n = total number of points (with extremities)
    n = 100
    ds = 1 / (n - 1)
    x = np.linspace(0, 1, n)

    ### solve y" = 0 with BC, equation 1 (E1) ###

    t1 = [1, 0, 0, 0]
    t2 = [0, 1, 0, 1]
    t3 = [0, 1, 0, 2]
    t4 = [1, 0, 0, 1]
    function1 = exact_solutionE1(x)
    A1 = fourth_derivative(n, ds)
    BC1, rhs1 = boundary_condition(ds, n, t1, t2, t3, t4, False)
    sol1 = sp.sparse.linalg.spsolve(A1 + BC1, rhs1)

    plt.plot(x, function1, color="blue", label="exact")
    plt.plot(x, sol1, color="orange", label="approx")
    plt.legend()
    plt.show()

    # ### solve y"" - y" = 0 with BC, equation 2 (E2) ###

    t1 = [1, 0, 0, 0]
    t2 = [0, 0, 1, 1]
    t3 = [1, 0, 0, 0]
    t4 = [0, 0, 1, 0]
    function2 = exact_solutionE2(x)
    A24 = fourth_derivative(n, ds)
    BC2, rhs2 = boundary_condition(ds, n, t1, t2, t3, t4, True)
    A22 = second_derivative(n, ds)
    A = A24 - A22 + BC2 
    sol2 = sp.sparse.linalg.spsolve(A, rhs2)

    plt.plot(x, function2, color="blue", label="exact")
    plt.plot(x, sol2, color="orange", label="approx")
    plt.legend()
    plt.show()


