from htstabilizer.stabilizer_circuits import *


"""
Example: Get a readout circuit for a 3-qubit stabilizer group
optimized for linear connectivity.

Hint: try 
    >>> from htstabilizer.connectivity_support import *
    >>> get_available_connectivities()
to get a list of all available connectivities and for instance
    >>> get_connectivity_graph(5, "T").draw()
to display the graph of a specific connectivity.
"""
qc = get_readout_circuit(Stabilizer(["XZZ", "ZXI", "ZIX"]), "linear")


"""
Notes on building the `Stabilizer` object
"""


# You can also pass a QuantumCircuit that only contains Clifford gates:
qc = QuantumCircuit(3)
# ... add gates
stabilizer = Stabilizer(qc, validate=True)

# If you specify `validate=True`, then the stabilizer will check
# that the given Paulis actually form a valid stabilizer:
stabilizer = Stabilizer(["XZZ", "ZXI", "ZIX"], validate=True)

# You can initialize a Stabilizer through a graph representing
# a graph state:
graph = Graph(3)
graph.add_path([0, 1, 2])
stabilizer = Stabilizer(graph)


# ... or by passing X and Z matrices for the stabilizer
import numpy as np
R = np.array([[1, 0, 0],
              [0, 1, 0],  # each column corresponds to one Pauli
              [0, 0, 1]])
S = np.array([[0, 1, 1],
              [1, 0, 0],
              [1, 0, 0]])
s3 = Stabilizer((R, S))

# This would be equivalent to `Stabilizer(["XZZ", "ZXI", "ZIX"])`
