

from typing import List, Literal, Tuple
from .graph import Graph


SupportedConnectivity = Literal["all", "linear", "star", "cycle", "T", "Q", "E", "H", "ladder"]


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
        (6, "all"),
        (6, "linear"),
        (6, "star"),
        (6, "ladder"),
        (6, "E"),
        (6, "H"),
        (6, "Q"),
    ]


def is_connectivity_supported(num_qubits: int, connectivity: SupportedConnectivity):
    try:
        assert_connectivity_is_supported(num_qubits, connectivity)
        return True
    except AssertionError:
        return False


def assert_connectivity_is_supported(num_qubits: int, connectivity: SupportedConnectivity):
    if num_qubits < 2 or num_qubits > 6:
        raise AssertionError(f"The given stabilizer has {num_qubits} qubits which is not supported.")

    if num_qubits == 2 and connectivity != "all":
        raise AssertionError(f"The connectivity '{connectivity}' is not supported for 2 qubits, did you mean 'all'?")

    is_supported = (num_qubits == 2 and connectivity == "all") or \
        (num_qubits == 3 and connectivity in ["all", "linear"]) or \
        (num_qubits == 4 and connectivity in ["all", "linear", "star", "cycle"]) or \
        (num_qubits == 5 and connectivity in ["all", "linear", "star", "cycle", "T",  "Q"]) or \
        (num_qubits == 6 and connectivity in ["all", "linear", "star", "ladder", "E", "H", "Q"])
    if not is_supported:
        raise AssertionError(f"The connectivity '{connectivity}' is not supported for {num_qubits} qubits.")


def get_connectivity_graph(
        num_qubits: int,
        connectivity: Literal["all", "linear", "star", "cycle", "T", "Q", "ladder", "E", "H"]
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
        assert num_qubits in [5, 6], "Q connectivity is only possible for 5 and 6 qubits"
        graph = Graph.linear(num_qubits)
        graph.add_edge(num_qubits - 1, num_qubits - 4)
        return graph
    if connectivity == "star":
        return Graph.star(num_qubits)
    if connectivity == "cycle":
        assert num_qubits in [4, 5], "Cycle connectivity is available for 4 and 5 qubits"
        return Graph.cycle(num_qubits)
    if connectivity == "ladder":
        assert num_qubits in [6], "Ladder connectivity is available for 6 qubits"
        graph = Graph.cycle(num_qubits)
        graph.add_edge(1, 4)
        return graph
    if connectivity == "E":
        assert num_qubits in [6], "E connectivity is available for 6 qubits"
        graph = Graph(num_qubits)
        graph.add_path([3, 0, 1, 2, 5])
        graph.add_edge(1, 4)
        return graph
    if connectivity == "H":
        assert num_qubits in [6], "E connectivity is available for 6 qubits"
        graph = Graph(num_qubits)
        graph.add_path([0, 1, 2])
        graph.add_path([3, 4, 5])
        graph.add_edge(1, 4)
        return graph
    
    assert False, f"Unsupported connnectivity: {connectivity}"
