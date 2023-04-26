try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import data  # relative-import the *package* containing the templates
from qiskit import QuantumCircuit

"""Read quantum circuits from lookup files



File format specification
-------------------------

Filename: 

    [circuit collection type][qubit number]-[connectivity].txt

where [circuit collection type] takes the values "stabilizer" or "mub"
and connectivity may be for example be "linear", "star", "T", "cycle" ...
    
Example: stabilizer3-linear.txt
Each line contains a circuit


Stabilizer circuit files
------------------------
A stabilizer circuit file for n qubits contains one circuit for each 
LC equivalence class with the index being the class id as specified
in the LCClass class. 
  Additionally, the exact graph that is implemented by the circuit
is specified as there are (in most cases) many different graphs in 
one class. 
  Stabilizer circuit descriptions are formatted in the following way:

  [graph-id]:[cost]:[depth]:[circuit-specification]

[graph-id]: integer, Contains the graph in compressed form as a bitstring integer
            encoding one half of the adjacency matrix beginning with the 
            least significant bit. For example take a 5×5 adjacency matrix,
            then the bit positions for each entry in the upper half are
                           - 0 1 2 3
                             - 4 5 6
                               - 7 8
                                 - 9
                                   -
[cost]:     integer, number of native 2-qubit gates necessary for the circuit
[depth]:    integer, circuit depth in terms of native 2-qubit gates
[circuit-specification]: string

            A circuit specification has the following structure

                [gate identifier][qubit1(optional:","[qubit2])]" "...

            where valid gate identifiers are 
                "h":    Hadamard gate
                "s":    Phase gate
                "hs":   Phase gate followed by a Hadamard gate (inverse order because mathematical operator evaluation is from right to left)
                "sh":   Hadamard gate followed by an sh gate
                "hsh":  Consecutive execution of Hadamard, Phase and again Hadamard gate
                "cx":   Controlled X gate
                "cz":   Controlled Z gate
                "swap": Swap gate
            The gate identifier is then followed by one or two qubit registers
            starting at zero. In the case of 2 qubits (only applies to the CX and CZ gate), 
            they separated by a comma ",". Here, the first qubit is the target qubit and the second
            the control qubit. 
            Example:

                h1 s3 cx0,2 swap1,2 h3 h0

"""

file_cache = {}


class CircuitInfo:
    def __init__(self, num_qubits: int, line: str):
        components = line.split(":")
        assert len(components) == 4
        self.num_qubits = num_qubits
        self.graph_id = int(components[0])
        self.cost = int(components[1])
        self.depth = int(components[2])
        self.circuit_string = components[3]

    def parse_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        instructions = self.circuit_string.split(" ")
        for instruction in instructions:
            if len(instruction) == 0:
                continue
            if instruction[0] == 'c':
                qubits = [int(qubit) for qubit in instruction[2:].split(",")]
                assert len(qubits) == 2, "Invalid 2-Qubit instruction specifcation"
                if instruction[1] == 'x':
                    qc.cx(qubits[0], qubits[1])
                elif instruction[1] == 'z':
                    qc.cz(qubits[0], qubits[1])
                else:
                    assert False, "Invalid instruction name"
            elif instruction[0] == 'h':
                if instruction[1] == 's':
                    pass
                else:
                    qc.h(int(instruction[1:]))
                pass
            elif instruction[0] == 's':
                if instruction[1] == 'w':
                    qubits = [int(qubit) for qubit in instruction[4:].split(",")]
                    qc.swap(qubits[0], qubits[1])
                else:
                    qc.s(int(instruction[1:]))
                pass
            else:
                assert False, "Invalid instruction name"
        qc.depth()
        return qc


def circuit_lookup(num_qubits: int, connectivity: str, id):
    filename = f"stabilizer{num_qubits}-{connectivity}.txt"

    try:
        circuitInfos = file_cache[filename]
    except KeyError:
        lines = pkg_resources.read_text(data, filename).split("\n")
        circuitInfos = [CircuitInfo(num_qubits, line) for line in filter(lambda x: len(x) != 0, lines)]
        file_cache[filename] = circuitInfos

    return circuitInfos[id]
