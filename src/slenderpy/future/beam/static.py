import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt

import fd_utils as FD 

def clean_matrix(n : int, remove):
    # renvoie une matrice qui permet d'éliminer les termes polluants les BC 
    res = sp.sparse.lil_matrix((n,n))
    remove = remove.todense()

    res[1, 0] -= remove[1,0]
    res[1, 1] -= remove[1,1]
    res[1, 2] -= remove[1,2]
    res[-2, -1] -= remove[-2,-1]
    res[-2, -2] -= remove[-2,-2]
    res[-2, -3] -= remove[-2,-3]

    return res

def solve_beam_const(n : int, ds : float, EI: float, H : float, bc:FD.BoundaryCondition):
    ### equation : EI*(d^4/dx^4)*y - H*(d^2/dx^2)*y = F(x) ###
    A4 = EI*FD.fourth_derivative(n, ds)
    A2 = -H*FD.second_derivative(n, ds) # si H = 0 D2 est quand même calculé ? 
    BC, rhs = bc.compute(ds, n)
    CM = clean_matrix(n, A2)
    A = A4 + A2 + BC + CM
    sol = sp.sparse.linalg.spsolve(A, rhs)
    return sol


def exact_solutionE1(x):
    return x**3 - x**2 + x



def exact_solutionE2(x):
    A = -1 / (np.exp(1) ** 2 - 1)
    B = np.exp(1) ** 2 / (np.exp(1) ** 2 - 1)
    D = -B - A
    C = -D - A * np.exp(1) - B * np.exp(-1)
    return A * np.exp(x) + B * np.exp(-x) + C * x + D

if __name__ == "__main__":
    # n = total number of points (with extremities)
    n = 100
    ds = 1 / (n - 1)
    x = np.linspace(0, 1, n)

    ### solve y"" = 0 with BC, equation 1 (E1) ###

    left = [[1, 0, 0, 0], [0, 1, 0, 1]]
    right = [[0, 1, 0, 2], [1, 0, 0, 1]] 
    EI = 1
    H = 1

    bc = FD.BoundaryCondition(4, left, right)
    function1 = exact_solutionE1(x)
    sol1 = solve_beam_const(n,ds,EI,H,bc)

    plt.plot(x, function1, color="blue", label="exact")
    plt.plot(x, sol1, color="orange", label="approx")
    plt.legend()
    plt.show()

    # # ### solve y"" - y" = 0 with BC, equation 2 (E2) ###

    left = [[1, 0, 0, 0], [0, 0, 1, 1]]
    right = [[1, 0, 0, 0],[0, 0, 1, 0]]
    bc = FD.BoundaryCondition(4,left,right)
    function2 = exact_solutionE2(x)
    sol2 = solve_beam_const(n,ds,EI,H,bc)

    plt.plot(x, function2, color="blue", label="exact")
    plt.plot(x, sol2, color="orange", label="approx")
    plt.legend()
    plt.show()

