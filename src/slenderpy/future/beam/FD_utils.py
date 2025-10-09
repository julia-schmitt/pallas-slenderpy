from typing import Tuple, Optional

import numpy as np
import scipy as sp 


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



class BoundaryCondition:
    """Object to deal with boundary conditions."""

    def __init__(
        self,
        t1: Optional[Tuple[float, float, float, float]] = None,
        t2: Optional[Tuple[float, float, float, float]] = None,
        t3: Optional[Tuple[float, float, float, float]] = None,
        t4: Optional[Tuple[float, float, float, float]] = None,
        # pos: Optional[str] = None,
    ) -> None:
        
        if t1 is None:
            t1 = (1.0, 0.0, 0.0, 0.0)
        if t2 is None:
            t2 = (0.0, 1.0, 0.0, 0.0)
        if (not isinstance(t1, (tuple, list))) or (not isinstance(t2, (tuple, list))):
            raise TypeError("Inputs t1 and t2 must be list or tuples")
        if len(t1) != 4 or len(t2) != 4:
            raise ValueError("Inputs t1 and t2 must have 4 elements")
        # if pos == "left":
        #     self.pp = -1.0
        # elif pos == "right":
        #     self.pp = +1.0
        # else:
        #     raise ValueError("Input pos must be either 'left' or 'right'")
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4

    def compute(self, ds: float, n:int, remove: np.ndarray[[float], [float]])-> Tuple[np.ndarray[[float], [float]], np.ndarray[float]]:
        a1, b1, c1, d1 = self.t1
        a2, b2, c2, d2 = self.t2
        a3, b3, c3, d3 = self.t3
        a4, b4, c4, d4 = self.t4

        rankL = np.linalg.matrix_rank(np.array([self.t1, self.t2]))
        if rankL < 2:
            raise ValueError("The left boundary conditions are not linearly independant")
        
        rankR = np.linalg.matrix_rank(np.array([self.t3, self.t4]))
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

        remove = remove.todense()

        bc_matrix[1, 0] -= remove[1,0]
        bc_matrix[1, 1] -= remove[1,1]
        bc_matrix[1, 2] -= remove[1,2]
        bc_matrix[-2, -1] -= remove[-2,-1]
        bc_matrix[-2, -2] -= remove[-2,-2]
        bc_matrix[-2, -3] -= remove[-2,-3]

        bc_matrix = sp.sparse.bsr_matrix(bc_matrix)

        rhs = np.zeros(n)
        rhs[0] = d1
        rhs[1] = d2
        rhs[-1] = d4
        rhs[-2] = d3

        return bc_matrix, rhs

