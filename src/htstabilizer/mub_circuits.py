
from .connectivity_support import assert_connectivity_is_supported, is_connectivity_supported
from . import circuit_lookup

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
    num_qubits : int
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

    assert_connectivity_is_supported(num_qubits, connectivity)
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
    num_qubits : int
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

    assert_connectivity_is_supported(num_qubits, connectivity)
    return circuit_lookup.mub_circuit_lookup(num_qubits, connectivity).mubs


def get_mub_info(
        num_qubits: int,
        connectivity: Literal["all", "linear", "star", "cycle", "T",  "Q"]
) -> dict:
    """Get info about a set of MUB circuits (e.g. for comparing different connectivities 
    and choosing the best among a possible selection). This function returns a dictionary 
    containing values for the following keys:
        - `"num circuits"`:            number of MUB circuits (2^n+1)
        - `"max two-qubit count"`:     the maximum number of native two-qubit gates (cx or cz, swap=3cx) 
                                       one of the circuits has.
        - `"max two-qubit depth"`:     the maximum depth of native two-qubit gates one of the circuits has.
        - `"average two-qubit gates"`: the average number of native two-qubit gates across all circuits

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    connectivity : Literal["all", "linear", "star", "cycle", "T", "Q"]
        Connectivity type

    Returns
    -------
    dict
        A dictonary containing info about the MUB

    Raises
    ------
    ValueError
        Raised if the connectivity is not supported
    """

    assert_connectivity_is_supported(num_qubits, connectivity)

    mub_info = circuit_lookup.mub_circuit_lookup(num_qubits, connectivity)
    info = {}
    info["num circuits"] = mub_info.num_qubits**2 + 1
    info["max two-qubit count"] = mub_info.max_cost
    info["max two-qubit depth"] = mub_info.max_depth
    info["average two-qubit gates"] = mub_info.total_cost / info["num circuits"]
    return info
