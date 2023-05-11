

from typing import List, Literal, Tuple
from .graph import Graph


SupportedConnectivity = Literal["all", "linear", "star", "cycle", "T", "Q"]


def get_available_connectivities() -> List[Tuple[int, str]]:
    """
    Get a list of available connectivities in form of tuples
    with number of qubits and connectivity name, e.g. (4, "linear")
    """
    return [
        (2, "all"),
        (3, "all"),
        (3, "linear"),
        (4, "all"),
        (4, "linear"),
        (4, "star"),
        (4, "cycle"),
        (5, "all"),
        (5, "linear"),
        (5, "star"),
        (5, "cycle"),
        (5, "T"),
        (5, "Q"),
    ]


def is_connectivity_supported(num_qubits: int, connectivity: SupportedConnectivity):
    if num_qubits < 2 or num_qubits > 5:
        return False
        raise ValueError(f"The given stabilizer has {num_qubits} qubits which is not supported.")

    return (num_qubits == 2 and connectivity == "all") or \
        (num_qubits == 3 and connectivity in ["all", "linear"]) or \
        (num_qubits == 4 and connectivity in ["all", "linear", "star", "cycle"]) or \
        (num_qubits == 5 and connectivity in ["all", "linear", "star", "cycle", "T",  "Q"])


def assert_connectivity_is_supported(num_qubits: int, connectivity: SupportedConnectivity):
    if not is_connectivity_supported(num_qubits, connectivity):
        raise ValueError(f"The connectivity {connectivity} is not valid/supported for {num_qubits} qubits.")


def get_connectivity_graph(
        num_qubits: int,
        connectivity: Literal["all", "linear", "star", "cycle", "T", "Q"]
) -> Graph:
    """
    Get a graph object for a given connectivity type. When drawn, the zeroth
    qubit is drawn at the topmost position and from there the order is clockwise. 

    Parameters
    ----------
    num_qubits : 
        Number of qubits
    connectivity : Literal["all", "linear", "star", "cycle", "T", "Q"]
        Connectivity type

    Returns
    -------
    Graph
        Graph instance
    """
    assert_connectivity_is_supported(num_qubits, connectivity)

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
