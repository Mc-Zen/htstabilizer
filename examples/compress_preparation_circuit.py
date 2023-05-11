
from htstabilizer.stabilizer_circuits import *
from qiskit.quantum_info import random_clifford

"""
You can compress a Clifford circuit by replacing it with a
hardware-tailored optimized circuit. 

The resulting circuit will only contain two-qubit gates between 
connected qubits (with respect to the given connectivity), so no 
further SWAP gates will be introduced when transpiling the circuit
to hardware connectivity. 

Hint: try 
    >>> from htstabilizer.connectivity_support import *
    >>> get_available_connectivities()
to get a list of all available connectivities and for instance
    >>> get_connectivity_graph(5, "T").draw()
to display the graph of a specific connectivity.
"""


# Generate some Clifford circuit
qc = random_clifford(5, seed=23).to_circuit()

# Optimizing it for T-shaped connectivity
compressed_qc = compress_preparation_circuit(qc, "T")
print(qc)
print(compressed_qc)
