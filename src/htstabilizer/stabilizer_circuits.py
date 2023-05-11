
from .connectivity_support import assert_connectivity_is_supported
from .stabilizer import Stabilizer
from . import circuit_lookup, lc_classes, find_local_clifford_layer
from .graph import Graph
from .find_local_clifford_layer import find_local_clifford_layer, local_clifford_layer_to_circuit
from .rotate_stabilizer_into_state import rotate_stabilizer_into_state

from qiskit import QuantumCircuit
from typing import Literal


def get_preparation_circuit(
        stabilizer: Stabilizer,
        connectivity: Literal["all", "linear", "star", "cycle", "T",  "Q"] = "all"
) -> QuantumCircuit:
    """
    Get an optimal, hardware-tailored preparation circuit to prepare a stabilizer
    state with given stabilizer group on a certain hardware connectivity. 

    Note: Phases of the Paulis in the stabilizer group are not taken into account. 
    You can use ``compress_preparation_circuit()`` to get an optimal preparation
    circuit for a specific stabilizer state from a clifford circuit. 

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

    Example
    -------

    >>> qc = get_preparation_circuit(Stabilizer(["XZZ", "ZXI", "ZIX"]), "linear")


    Hint: use :func:`connectivity_support.get_connectivity_graph(num_qubits, connectivity)` to 
    get a graph instance that can be visualized. 

    Parameters
    ----------
    stabilizer : Stabilizer
        Stabilizer state
    connectivity : Literal["all", "linear", "star", "cycle", "T", "Q"]
        Connectivity type

    Returns
    -------
    QuantumCircuit
        Preparation circuit for input stabilizer
    """

    assert_connectivity_is_supported(stabilizer.num_qubits, connectivity)

    lc_class_id = lc_classes.determine_lc_class(stabilizer).id()
    circuit_info = circuit_lookup.stabilizer_circuit_lookup(stabilizer.num_qubits, connectivity, lc_class_id)
    layer = find_local_clifford_layer(stabilizer.R, stabilizer.S, Graph.decompress(stabilizer.num_qubits, circuit_info.graph_id))
    if layer is None:
        raise RuntimeError("No circuit could be found. Please validate the input stabilizer.")
    layer_circuit = local_clifford_layer_to_circuit(layer).inverse()
    return circuit_info.parse_circuit().compose(layer_circuit)  # type: ignore


def get_readout_circuit(
        stabilizer: Stabilizer,
        connectivity: Literal["all", "linear", "star", "cycle", "T", "Q"]
) -> QuantumCircuit:
    """
    Get an optimal, hardware-tailored readout (diagonalization) circuit
    for given stabilizer state on a certain hardware connectivity. Look at 
    the documentation of `get_preparation_circuit()` for more information. 

    Parameters
    ----------
    stabilizer : Stabilizer
        Stabilizer state
    connectivity : Literal["all", "linear", "star", "cycle", "T", "Q"]
        Connectivity type

    Returns
    -------
    QuantumCircuit
        Preparation circuit for input stabilizer
    """

    assert_connectivity_is_supported(stabilizer.num_qubits, connectivity)
    return get_preparation_circuit(stabilizer, connectivity).inverse()


def compress_preparation_circuit(
        circuit: QuantumCircuit,
        connectivity: Literal["all", "linear", "star", "cycle", "T",  "Q"] = "all"
) -> QuantumCircuit:
    """
    Optimize a Clifford preparation circuit, i.e., a circuit that is placed at the
    beginning of a computation - starting with the all-zero state. The circuit can 
    only contain I, X, Y, Z, H, S, SDG, CX, CZ and SWAP gates. 
    
    Look at the documentation of `get_preparation_circuit()` for more information. 

    Parameters
    ----------
    circuit : QuantumCircuit
        Clifford preparation circuit
    connectivity : Literal["all", "linear", "star", "cycle", "T",  "Q"]
        Connectivity type


    Returns
    -------
    QuantumCircuit
        Optimized circuit
    """

    assert_connectivity_is_supported(circuit.num_qubits, connectivity)
    optimized_circuit = get_preparation_circuit(Stabilizer(circuit), connectivity)
    return rotate_stabilizer_into_state(optimized_circuit, circuit, inplace=True)
