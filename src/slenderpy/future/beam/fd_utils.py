from typing import Tuple, Optional

import numpy as np
import scipy as sp


def first_derivative(n: int, ds: float) -> sp.sparse.spmatrix:
    # centered scheme, the first and last line have to be completed with BC (order 2)

    dinf = -1.0 * np.ones((n - 1,)) / (2 * ds)
    dsup = +1.0 * np.ones((n - 1,)) / (2 * ds)

    dinf[-1] = 0
    dsup[0] = 0

    res = sp.sparse.diags([dinf, dsup], [-1, 1])

    return res


def second_derivative(n: int, ds: float) -> sp.sparse.spmatrix:
    # centered scheme, the first and last line have to be completed with BC (order 2)

    dinf = +1.0 * np.ones((n - 1,)) / ds**2
    diag = -2.0 * np.ones((n,)) / ds**2
    dsup = +1.0 * np.ones((n - 1,)) / ds**2

    dinf[-1] = 0
    diag[0] = 0
    diag[-1] = 0
    dsup[0] = 0

    res = sp.sparse.diags([dinf, diag, dsup], [-1, 0, 1])

    return res


def fourth_derivative(n: int, ds: float) -> sp.sparse.spmatrix:
    # centered scheme, the two first and two last line have to be completed with BC (order 4)

    dinf2 = +1.0 * np.ones((n - 2)) / ds**4
    dinf1 = -4.0 * np.ones((n - 1,)) / ds**4
    diag = +6.0 * np.ones((n,)) / ds**4
    dsup1 = -4.0 * np.ones((n - 1,)) / ds**4
    dsup2 = +1.0 * np.ones((n - 2,)) / ds**4

    dinf2[-1] = 0
    dinf2[-2] = 0
    dinf1[-1] = 0
    dinf1[-2] = 0
    dinf1[0] = 0
    diag[0] = 0
    diag[1] = 0
    diag[-2] = 0
    diag[-1] = 0
    dsup1[0] = 0
    dsup1[1] = 0
    dsup1[-1] = 0
    dsup2[0] = 0
    dsup2[1] = 0

    res = sp.sparse.diags([dinf2, dinf1, diag, dsup1, dsup2], [-2, -1, 0, 1, 2])

    return res


class BoundaryCondition:
    """Object to deal with boundary conditions."""

    def __init__(
        self,
        order: int,  # pour l'instant je traite que le cas 2 et 4
        left: Optional[
            Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]
        ] = None,
        right: Optional[
            Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]
        ] = None,
    ) -> None:

        if order == 4:
            if left is None:
                left = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))

            if right is None:
                right = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))

            if (not isinstance(left, (tuple, list))) or (
                not isinstance(right, (tuple, list))
            ):
                raise TypeError("Inputs left and right must be list or tuples")

            if len(left) != 2 or len(right) != 2:
                raise ValueError("Need two boundary condition for each extremity")

            if (
                len(left[0]) != 4
                or len(left[1]) != 4
                or len(right[0]) != 4
                or len(right[1]) != 4
            ):
                raise ValueError(
                    "Inputs must have 4 elements for each boundary condition"
                )

            rankL = np.linalg.matrix_rank(left)
            rankR = np.linalg.matrix_rank(right)

            if rankL < 2:
                raise ValueError(
                    "The left boundary conditions are not linearly independant"
                )

            if rankR < 2:
                raise ValueError(
                    "The right boundary conditions are not linearly independant"
                )

        if order == 2:
            if left is None:
                left = ((1.0, 0.0, 0.0, 0.0),)

            if right is None:
                right = ((1.0, 0.0, 0.0, 0.0),)

            if (not isinstance(left, (tuple, list))) or (
                not isinstance(right, (tuple, list))
            ):
                raise TypeError("Inputs left and right must be list or tuples")

            if len(left) != 1 or len(right) != 1:
                raise ValueError("Need one boundary condition for each extremity")

            if len(left[0]) != 4 or len(right[0]) != 4:
                raise ValueError(
                    "Inputs must have 4 elements for each boundary condition"
                )

            rankL = np.linalg.matrix_rank(left)
            rankR = np.linalg.matrix_rank(right)

            if rankL < 1:
                raise ValueError("There is no left boundary condition")

            if rankR < 1:
                raise ValueError("There is no right boundary condition")

        self.left = left
        self.right = right
        self.order = order

    def compute(
        self, ds: float, n: int
    ) -> Tuple[sp.sparse.spmatrix, np.ndarray[float]]:
        a1, b1, c1, d1 = self.left[0]
        a4, b4, c4, d4 = self.right[0]

        bc_matrix = sp.sparse.lil_matrix((n, n))
        rhs = np.zeros(n)

        bc_matrix[0, 0] = a1 - b1 / ds + c1 / ds**2
        bc_matrix[0, 1] = b1 / ds - 2 * c1 / ds**2
        bc_matrix[0, 2] = c1 / ds**2

        bc_matrix[-1, -1] = a4 + b4 / ds + c4 / ds**2
        bc_matrix[-1, -2] = -b4 / ds - 2 * c4 / ds**2
        bc_matrix[-1, -3] = c4 / ds**2

        rhs[0] = d1
        rhs[-1] = d4

        if self.order == 4:
            a2, b2, c2, d2 = self.left[1]
            a3, b3, c3, d3 = self.right[1]

            bc_matrix[1, 0] = a2 - b2 / ds + c2 / ds**2
            bc_matrix[1, 1] = b2 / ds - 2 * c2 / ds**2
            bc_matrix[1, 2] = c2 / ds**2

            bc_matrix[-2, -1] = a3 + b3 / ds + c3 / ds**2
            bc_matrix[-2, -2] = -b3 / ds - 2 * c3 / ds**2
            bc_matrix[-2, -3] = c3 / ds**2

            rhs[-2] = d3
            rhs[1] = d2

        return bc_matrix, rhs
