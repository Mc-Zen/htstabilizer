import numpy as np
from src.htstabilizer import f2_algebra as f2
from src.htstabilizer.find_local_clifford_layer import generate_local_clifford_symplectic_from_id
from src.htstabilizer.graph import Graph
from src.htstabilizer.stabilizer import Stabilizer


def random_stabilizer(graph: Graph) -> Stabilizer:
    """Create a random stabilizer that is local-Clifford equivalent to given
    graph state by applying a random single-qubit Clifford on each qubit. 

    Parameters
    ----------
    graph : Graph
        Graph describing a graph state

    Returns
    -------
    Stabilizer
        random stabilizer
    """
    stabilizer = Stabilizer(graph)
    A = generate_local_clifford_symplectic_from_id(np.random.randint(0, 6, graph.num_vertices))
    R_prime = f2.add(f2.mat_mul(A[0], stabilizer.R), f2.mat_mul(A[1], stabilizer.S))
    S_prime = f2.add(f2.mat_mul(A[2], stabilizer.R), f2.mat_mul(A[3], stabilizer.S))
    stabilizer.R = R_prime
    stabilizer.S = S_prime
    return stabilizer