"""Linear algebra over the binary field GF2"""

import numpy as np
import copy
from typing import Tuple, List


def mat_mul(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Multiply matrices m1 and m2 under modulo 2 arithmetic. """
    return (m1 @ m2) & 1

def add(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Add matrices m1 and m2 under modulo 2 arithmetic. """
    return m1 ^ m2


def rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Compute the reduced-row echelon form of a matrix A 
    with entries in the binary field GF(2). 

    Parameters
    ----------
    A : np.ndarray
        Input matrix, should only contain elements 0 or 1 as integers. 

    Returns
    -------
    Tuple[np.ndarray, List[int]]
        RREF of input matrix A as well as column indices of the pivot elements. 
    """
    m, n = A.shape  # m: rows, n: cols
    A = A % 2
    pivot_cols = []
    h = 0
    k = 0
    while h < m and k < n:
        found = False
        i = h
        while not found and i < m:
            if A[i, k] == 1:
                found = True
                break
            i += 1
        if not found:
            k += 1
        else:
            pivot_cols.append(k)
            temp = copy.deepcopy(A[h, :])
            A[h, :] = A[i, :]
            A[i, :] = temp
            for i in list(range(h)) + list(range(h+1, m)):
                A[i, :] = (A[i, :] + A[i, k]*A[h, :]) % 2
            h += 1
            k += 1
    return [A, pivot_cols]


def trf_swap_rows(i: int, j: int, m: int, dtype=np.int8) -> np.ndarray:
    """Get matrix T that swaps rows i,j of an m×n matrix A when 
    multiplied to from the left, TA. 

    Parameters
    ----------
    i,j   : int
            Indices of rows to swap in m×n matrix A
    m     : int
            Number of rows of matrix A
    dtype : type
            Data type of returned matrix, defaults to np.int8

    Returns
    -------
    np.ndarray
        m×m matrix T
    """
    out = np.eye(m, dtype=dtype)
    out[i, i] = 0
    out[j, j] = 0
    out[i, j] = 1
    out[j, i] = 1
    return out


def trf_add_row(i: int, j: int, m: int, dtype=np.int8) -> np.ndarray:
    """Get matrix T that adds row i to row j of an m×n matrix A when 
    multiplied to from the left, TA. 

    Parameters
    ----------
    i,j   : int
            Indices of rows to swap in m×n matrix A
    m     : int
            Number of rows of matrix A
    dtype : type
            Data type of returned matrix, defaults to np.int8

    Returns
    -------
    np.ndarray
        m×m matrix T
    """
    out = np.eye(m, dtype=dtype)
    out[j, i] = 1
    return out


def rank(A: np.ndarray) -> int:
    """ Computes the rank of a matrix A with coefficients in GF(2).
    """
    return len(rref(A)[1])


def null_space(A: np.ndarray) -> np.ndarray:
    """Compute the null space (kernel) of a matrix A
    with entries in the binary field GF(2). 

    Parameters
    ----------
    A : np.ndarray
        Input matrix

    Returns
    -------
    np.ndarray
        Null space of A
    """
    A_rref, pivot_cols = rref(A)
    cols = A.shape[1]

    out = []
    for i in range(cols):
        if i not in pivot_cols:  # loop through non-pivot columns
            vec = np.zeros(cols, dtype=np.int8)
            vec[i] = 1
            k = 0
            for j in pivot_cols:
                vec[j] = A_rref[k, i]
                k += 1
            out.append(vec % 2)
    return np.array(out)
