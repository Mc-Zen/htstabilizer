from qiskit import QuantumCircuit
from .stabilizer import Stabilizer
from . import circuit_lookup, lc_classes, find_local_clifford_layer
from .graph import Graph
from .find_local_clifford_layer import find_local_clifford_layer, local_clifford_layer_to_circuit
from typing import Literal


def get_preparation_circuit(stabilizer: Stabilizer,
                            connectivity: Literal["all", "linear", "T", "star", "cycle", "Q"]) \
        -> QuantumCircuit:
    """Get an optimal, hardware-tailored preparation circuit to prepare
    a given stabilizer state on a certain hardware connectivity. 

    Supported hardware connectivities:
    2 Qubits
        "all": Both qubits are connected
    3 Qubits
        "all": all-to-all connectivitiy
        "linear": Linear chain 0--1--2
    4 Qubits
        "all": all-to-all connectivitiy
        "linear": Linear chain 0--1--2--3
        "star": star-shaped connectivity, all connected to qubit 0, 0--{1,2,3}
        "cycle": cycle connectivity 0--1--2--3--0
    5 Qubits
        "all": all-to-all connectivitiy
        "linear": Linear chain 0--1--2--3--4
        "T": T-shaped connectivity 4--3--0--{1,2]}
        "star": star-shaped connectivity, all connected to qubit 0, 0--{1,2,3,4}
        "cycle": cycle connectivity 0--1--2--3--4--0
        "q": q-shaped connectivity, 0--1--2--3--4--1


  #  o---o
  #  |   |
  #  |   |
  #  o---o----o


    Hint: use get_connectivity_graph(num_qubits, connectivity) to 
    get a graph instance that can be visualized. 

    Parameters
    ----------
    stabilizer : Stabilizer
        Stabilizer state
    connectivity : Literal["all", "linear", "T", "star", "cycle", "Q"]
        Connectivity type

    Returns
    -------
    QuantumCircuit
        Preparation circuit for input stabilizer
    """
    n = stabilizer.num_qubits
    if n < 2 or n > 5:
        raise ValueError(f"The given stabilizer has {n} qubits which is not supported.")

    valid_connectivity = (n == 2 and connectivity == "all") or \
                         (n == 3 and connectivity in ["all", "linear"]) or \
                         (n == 4 and connectivity in ["all", "linear", "star", "cycle"]) or \
                         (n == 5 and connectivity in ["all", "linear", "star", "cycle", "T",  "Q"])
    if not valid_connectivity:
        raise ValueError(f"The connectivity {connectivity} is not valid/supported for {n} qubits.")

    lc_class_id = lc_classes.determine_lc_class(stabilizer).id()
    circuit_info = circuit_lookup.circuit_lookup(stabilizer.num_qubits, connectivity, lc_class_id)
    layer = find_local_clifford_layer(stabilizer.R, stabilizer.S, Graph.decompress(stabilizer.num_qubits, circuit_info.graph_id))
    if layer is None:
        raise RuntimeError("No circuit could be found. Please validate the input stabilizer.")
    layer_circuit = local_clifford_layer_to_circuit(layer).inverse()
    return circuit_info.parse_circuit().compose(layer_circuit)  # type: ignore


def get_readout_circuit(stabilizer: Stabilizer, connectivity: Literal["all", "linear", "T", "star"]):
    return get_preparation_circuit(stabilizer, connectivity).inverse()


def get_connectivity_graph(num_qubits: int, connectivity: Literal["all", "linear", "T", "star"]) -> Graph:
    """Get a graph object for a given connectivity type. When drawn, the zeroth
    qubit is drawn at the topmost position and from there the order is clockwise. 

    Parameters
    ----------
    num_qubits : 
        Number of qubits
    connectivity : Literal["all", "linear", "T", "star"]
        Connectivity type

    Returns
    -------
    Graph
        Graph instance
    """
    if connectivity == "all":
        return Graph.fully_connected(num_qubits)
    if connectivity == "linear":
        return Graph.linear(num_qubits)
    if connectivity == "T":
        assert num_qubits == 5, "T connectivity is only possible for 5 qubits"
        return Graph.pusteblume(num_qubits)
    if connectivity == "Q":
        assert num_qubits == 5, "Q connectivity is only possible for 5 qubits"
        graph = Graph(5)
        graph.add_path([0, 1, 2, 3, 4, 1])
        return graph
    if connectivity == "star":
        return Graph.star(num_qubits)
    if connectivity == "cycle":
        assert num_qubits in [4, 5], "Cycle connectivity is available for 4 and 5 qubits"
        return Graph.cycle(num_qubits)
    assert False, f"Unsupported connnectivity: {connectivity}"
