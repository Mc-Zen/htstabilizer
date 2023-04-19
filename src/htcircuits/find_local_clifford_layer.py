import itertools
from . import f2_algebra as f2
from .graph import Graph
import numpy as np
from typing import List, Tuple, Optional


def generate_local_clifford_symplectic(c: List[np.ndarray] | List[List[int]]) -> List[np.ndarray]:
    """Generate a Clifford A in symplectic form from a list of
    symplectic single-qubit Clifford matrices. A tuple of block 
    matrices Axx, Axz, Azx, Azz is returned, so that
        A = | Axx Axz |
            | Azx Azz |

    Valid single-qubit Cliffords are
     I    = [1,0,0,1]
     H    = [0,1,1,0]
     S    = [1,0,1,1]
     HS   = [1,1,1,0]
     SH   = [0,1,1,1]
     HSH  = [1,1,0,1]

    Example
    -------
      `generate_local_clifford_symplectic([[1,1,1,0], [0,1,1,0]])`

    to create a Clifford with HS on the first Qubit and H on the second. 

    Parameters
    ----------
    c : List[np.ndarray] | List[List[int]]
        List of single-qubit Cliffords in symplectic form. 

    Returns
    -------
    List[np.ndarray]
        Axx, Axz, Azx, Azz as a tuple
    """
    n = len(c)
    As = [np.zeros(shape=(n, n), dtype=np.int8) for i in range(4)]
    for i in range(n):
        for j in range(4):
            As[j][i, i] = c[i][j]
    return As


def generate_local_clifford_symplectic_from_id(clifford_ids: List[int]) -> List[np.ndarray]:
    """Similar to `generate_local_clifford_symplectic` but a list of
    Clifford "ids" ranging from 0 to 5 is expected. 
     0: I    = [1,0,0,1]
     1: H    = [0,1,1,0]
     2: S    = [1,0,1,1]
     3: HS   = [1,1,1,0]
     4: SH   = [0,1,1,1]
     5: HSH  = [1,1,0,1]

    Parameters
    ----------
    clifford_ids : List[int]
        List of clifford ids. 

    Returns
    -------
    List[np.ndarray]
        Axx, Axz, Azx, Azz as a tuple
    """
    cs = [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 1]]
    n = len(clifford_ids)
    As = [np.zeros(shape=(n, n), dtype=np.int8) for i in range(4)]
    for i in range(n):
        for j in range(4):
            As[j][i, i] = cs[clifford_ids[i]][j]
    return As


def generate_single_qubit_symplectic(c: np.ndarray | list, n: int, i: int):
    """
    Create a 2n×2n symplectic matrix in the form of 4 blocks

        A = | Axx Axz |
            | Azx Azz |

    where the ith diagonal entry of each block j contains the jth 
    entry of the given argument c. This is equivalent to a Clifford
    transform in symplectic form with single-qubit Clifford c on 
    qubit i. 

    Example
    -------
     `generate_single_qubit_symplectic([1,1,1,0], 2, 1)`

    gives

     Axx = | 0 0 |  Axz = | 0 0 |  Azx = | 0 0 |  Azz = | 0 0 |
           | 0 1 |        | 0 1 |        | 0 1 |        | 0 0 |


    Parameters
    ----------
    c : np.ndarray
        Array of 4 integers in {0,1} 
    n : int
        Size, i.e. number of qubits
    i : int
        Qubit

    Returns
    -------
    List[np.ndarray]
        A tuple of 4 n×n block matrices
    """
    As = [np.zeros(shape=(n, n), dtype=np.int8) for i in range(4)]
    for j in range(4):
        As[j][i, i] = c[j]
    return As


def check_LC(R: np.ndarray, S: np.ndarray, graph: Graph, A: List[np.ndarray]) -> bool:
    """Check that applying given Clifford A in symplectic form 
    to R, S rotates these into R', S' which are then diagonalized
    by uncomputing the graph state corresponding to the given graph. 

    Parameters
    ----------
    R : np.ndarray
        X components of the set of Paulis
    S : np.ndarray
        Z components of the set of Paulis
    graph : Graph
        Graph describing the graph state
    A : List[np.ndarray]
        Clifford in symplectic form given as block matrices [Axx, Axz, Azx, Azz]
        so that    A = | Axx Axz |
                       | Azx Azz |

    Returns
    -------
    bool
        True if A rotates R, S such that the result is diagonalized by the graph state
    """
    gamma = graph.adjacency_matrix
    Axx, Axz, Azx, Azz = A
    GAR = f2.mat_mul(f2.mat_mul(gamma, Axx), R)
    GAS = f2.mat_mul(f2.mat_mul(gamma, Axz), S)
    AR = f2.mat_mul(Azx, R)
    AS = f2.mat_mul(Azz, S)
    LHS = f2.add(f2.add(GAR, GAS), f2.add(AR, AS))
    return not np.any(LHS)


def find_local_clifford_layer(R: np.ndarray, S: np.ndarray, graph: Graph) -> Optional[List[np.ndarray]]:
    """Find a local Clifford that rotates a Pauli set P (given in its X and Z
    components by R and S respectively) into P' such that P' can be diagonalized
    (i.e. brought into the canonical computational basis) by uncomputing the 
    graph state |Γ〉 which is given by the graph argument. 

    Complexity
    ----------
    This algorithm is bounded by O(2^(2n)) although this case is actually
    not possible (I just don't have time enough to come up with the smallest
    bound). The average runtime however is much much less, because usually multiple
    solutions exist and also in practice the solution space is much less than 2n. 

    Parameters
    ----------
    R : np.ndarray
        X components of the set of Paulis
    S : np.ndarray
        Z components of the set of Paulis
    graph : Graph
        Graph that describes the graph state |Γ〉

    Returns
    -------
    Optional[List[np.ndarray]]
        Local Clifford in symplectic form given as block matrices [Axx, Axz, Azx, Azz]
        so that    A = | Axx Axz |
                       | Azx Azz |
        or None in the case such a Clifford does not exist, i.e. any extension of P into
        a stabilizer S corresponds to a stabilizer state |ψ〉 that is not local-Clifford 
        equivalent to |Γ〉. 
    """
    gamma = graph.adjacency_matrix
    assert gamma.shape[0] == gamma.shape[1] and R.shape == S.shape and R.shape[0] == gamma.shape[0]

    n = gamma.shape[0]
    m = R.shape[1]

    c_i = np.array([1, 0, 0, 1], dtype=np.int8)
    c_h = np.array([0, 1, 1, 0], dtype=np.int8)
    c_s = np.array([1, 0, 1, 1], dtype=np.int8)
    c_hs = np.array([1, 1, 1, 0], dtype=np.int8)
    cs = [c_i, c_hs, c_h, c_s]
    p = len(cs)

    Rs = np.zeros(shape=(p*n, n*m), dtype=np.int8)
    Ss = np.zeros(shape=(p*n, n*m), dtype=np.int8)

    for i in range(n):
        for j in range(p):
            Axx, Axz, Azx, Azz = generate_single_qubit_symplectic(cs[j], n, i)
            R_i = np.zeros(shape=R.shape, dtype=np.int8)
            R_i[i] = R[i]
            S_i = np.zeros(shape=S.shape, dtype=np.int8)
            S_i[i] = S[i]
            S_prime = f2.add(f2.mat_mul(Axx, R_i), f2.mat_mul(Axz, S_i))
            R_prime = f2.add(f2.mat_mul(gamma, S_prime), f2.add(f2.mat_mul(Azx, R_i), f2.mat_mul(Azz, S_i)))
            R_prime = R_prime.reshape(n*m)
            S_prime = S_prime.reshape(n*m)
            Rs[p*i+j] = R_prime
            Ss[p*i+j] = S_prime

    Rs = Rs.transpose()
    kernel = f2.null_space(Rs)
    rank = f2.rank(Rs)
    # print("Rank =", rank, ", N =\n", kernel)

    # O(2^(2n)) algorithm
    rank = kernel.shape[0]
    combinations = np.arange(2**rank)
    combinations = np.array([i for i in itertools.product([0, 1], repeat=rank)])
    cc = f2.mat_mul(combinations, np.array(kernel))

    for row in cc:
        for i in range(n):
            c1 = row[i*4+0] | row[i*4+1]
            c2 = row[i*4+2] | row[i*4+3]
            if c1 ^ c2 == 1:
                # print(row)
                Us = [np.zeros(4, dtype=np.int8) for j in range(n)]
                for j in range(n):
                    for k in range(4):
                        if row[j*4+k]:
                            Us[j] = f2.add(Us[j], np.array(cs[k]))
                As = generate_local_clifford_symplectic(Us)
                # print(Us, *As, sep="\n")

                return As
    return None
