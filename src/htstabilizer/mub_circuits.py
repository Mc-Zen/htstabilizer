
from .connectivity_support import is_connectivity_supported
from .stabilizer import Stabilizer
from . import circuit_lookup, lc_classes, find_local_clifford_layer
from .graph import Graph
from .find_local_clifford_layer import find_local_clifford_layer, local_clifford_layer_to_circuit

from qiskit import QuantumCircuit
from typing import List, Literal


def get_mub_circuits(
        num_qubits: int,
        connectivity: Literal["all", "linear", "star", "cycle", "T",  "Q"]
) -> List[QuantumCircuit]:
    """
    Get an optimal, hardware-tailored preparation collection of quantum circuits
    to perform full state tomography for a given hardware connectivity. 

    Supported hardware connectivities:
    - 2 Qubits
        - `"all"`: Both qubits are connected
    - 3 Qubits
        - `"all"`: all-to-all connectivitiy
        - `"linear"`: Linear chain 0--1--2
    - 4 Qubits
        - `"all"`: all-to-all connectivitiy
        - `"linear"`: Linear chain 0--1--2--3
        - `"star"`: star-shaped connectivity, all connected to qubit 0, 0--{1,2,3}
        - `"cycle"`: cycle connectivity 0--1--2--3--0
    - 5 Qubits
        - `"all"`: all-to-all connectivitiy
        - `"linear"`: Linear chain 0--1--2--3--4
        - `"star"`: star-shaped connectivity, all connected to qubit 0, 0--{1,2,3,4}
        - `"cycle"`: cycle connectivity 0--1--2--3--4--0
        - `"T"`: T-shaped connectivity 4--3--0--{1,2]}
        - `"Q"`: q-shaped connectivity, 0--1--2--3--4--1


    Hint: use :func:`connectivity_support.get_connectivity_graph(num_qubits, connectivity)` to 
    get a graph instance that can be visualized. 


    Parameters
    ----------
    num_qubits : 
        Number of qubits
    connectivity : Literal["all", "linear", "star", "cycle", "T", "Q"]
        Connectivity type

    Returns
    -------
    List[QuantumCircuit]
        List of quantum circuits

    Raises
    ------
    ValueError
        Raised if the connectivity is not supported
    """
    if not is_connectivity_supported(num_qubits, connectivity):
        raise ValueError(f"The connectivity {connectivity} is not valid/supported for {num_qubits} qubits.")

    return circuit_lookup.mub_circuit_lookup(num_qubits, connectivity).circuits


def get_mubs(
        num_qubits: int,
        connectivity: Literal["all", "linear", "star", "cycle", "T",  "Q"]
) -> List[List[str]]:
    """
    Get a collection of mutually unbiased bases in form of 2^n+1 sets
    each containing n Pauli strings (big-endian, the first character 
    in every Pauli string corresponds to the first qubit). These n Pauli
    strings make up a generator for each set. 

    All generators are mutually disjoint and together make up the entire
    Pauli group (except for the identity operator). 

    Parameters
    ----------
    num_qubits : 
        Number of qubits
    connectivity : Literal["all", "linear", "star", "cycle", "T", "Q"]
        Connectivity type

    Returns
    -------
    List[List[str]]
        2^n+1 lists with n Pauli strings each

    Raises
    ------
    ValueError
        Raised if the connectivity is not supported
    """

    if not is_connectivity_supported(num_qubits, connectivity):
        raise ValueError(f"The connectivity {connectivity} is not valid/supported for {num_qubits} qubits.")

    return circuit_lookup.mub_circuit_lookup(num_qubits, connectivity).mubs
