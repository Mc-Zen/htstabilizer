from typing import Dict, List, Literal, Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from .stabilizer import Stabilizer
from .stabilizer_circuits import get_readout_circuit
from qiskit.result import Result
from qiskit.quantum_info import Pauli
from qiskit.circuit import Qubit


class ReadoutInfo:
    def __init__(self,
                 readout_circuit: QuantumCircuit,
                 total_num_qubits: int,
                 measured_qubits: Optional[List[Qubit]] = None
                 ):
        """Create a readout info object storing the readout circuit, 
        measured qubits etc. 

        Parameters
        ----------
        readout_circuit : QuantumCircuit
            Quantum readout circuit
        total_num_qubits : int
            Number of qubits for the system (may be larger than 
            `readout_circuit.num_qubits` if only some qubits are measured)
        measured_qubits : Optional[List[Qubit]], optional
            The qubits that are measured or None if all are measured, 
            by default None
        """
        self.circuit = readout_circuit
        self.qubits = measured_qubits
        self.total_num_qubits = total_num_qubits


def stabilizer_measurement_circuit(
        preparation_circuit: QuantumCircuit,
        stabilizer: Stabilizer,
        connectivity: Literal["all", "linear", "star", "cycle", "T", "Q"] = "all",
        measured_qubits: Optional[Union[QuantumRegister, List[Qubit], List[int]]] = None,
) -> QuantumCircuit:
    """
    Return a circuit to measure a stabilizer of the quantum state prepared by given 
    preparation circuit. 

    Parameters
    ----------
    preparation_circuit : QuantumCircuit
        The preparation circuit for the state to be tomographed
    stabilizer : Stabilizer
        The stabilizer to measure
    connectivity : Literal["all", "linear", "star", "cycle", "T", "Q"]
        The hardware-connectivity to use in order to get an optimal readout circuit 
        (use `"all"` if hardware-connectivity does not matter)
    measured_qubits : Optional[QuantumRegister | List[Qubit]], optional
        The qubits to be measured, omit this to measure all qubits, by default None


    Example
    ------

    >>> smc = stabilizer_measurement_circuit(my_prep_circuit, Stabilizer(["XI", "IZ"]), "linear")

    It is also possible to measure only some qubits

    >>> qr = QuantumRegister(6)
    >>> bell = QuantumCircuit(qr)
    >>> bell.h(qr[3])
    >>> bell.cx(qr[3], qr[5])
    >>> smc = stabilizer_measurement_circuit(bell, Stabilizer(["XI", "IZ"]), "all", [qr[3], qr[5]]) 


    Returns
    -------
    QuantumCircuit
        A quantum circuit that prepares the state and measures it. 
    """

    if measured_qubits is None and preparation_circuit.num_qubits != stabilizer.num_qubits:
        raise ValueError("The number of qubits does not match for preparation circuit and stabilizer")
    elif measured_qubits is not None and stabilizer.num_qubits != len(measured_qubits):
        raise ValueError("The number of qubits to be measured does not match the number of qubits of the stabilizer given")
    # assert preparation_circuit.num_qubits == stabilizer.num_qubits, "The number of qubits does not match for preparation circuit and stabilizer"
    readout_circuit = get_readout_circuit(stabilizer, connectivity)
    circuit: QuantumCircuit = preparation_circuit.compose(readout_circuit, qubits=measured_qubits)  # type: ignore
    circuit.measure_all()
    if isinstance(measured_qubits, list):
        if isinstance(measured_qubits[0], int):
            measured_qubits = [circuit.qubits[i] for i in measured_qubits]
    if isinstance(measured_qubits, QuantumRegister):
        measured_qubits = [qubit for qubit in measured_qubits]
    if circuit.metadata is None:
        circuit.metadata = {}
    circuit.metadata["readout info"] = ReadoutInfo(readout_circuit, preparation_circuit.num_qubits, measured_qubits)
    return circuit


Bitstring = np.int64


class BinaryResult:
    """
    Class for storing a single quantum circuit result with result
    signature and count. 

    The result signature is encoded in a binary string in little-endian
    (least-significant bit represents the zeroth quantum bit result). 

    """

    __slots__ = ("bitstring", "count")

    def __init__(self, bitstring: Union[Bitstring, int], count: int):
        self.bitstring = Bitstring(bitstring)
        self.count = count

    def __eq__(self, other) -> bool:
        return self.bitstring == other.bitstring and self.count == other.count

    def __str__(self, num_qubits=0) -> str:
        return f"{self.bitstring:0>{num_qubits}b}: {self.count}"

    def __repr__(self) -> str:
        return f"BinaryResult({self.bitstring}, {self.count})"


class CircuitResult:
    """
    Class encapsuling results from one quantum circuit, similar to 
    :class:`qiskit.result.ExperimentResultData` but the keys are not stored
    using hexadecimal strings but binary bitstrings which allows for
    some performance improvements when evaluating readout results. . 

    """

    __slots__ = ("results", "num_qubits")

    def __init__(self, counts: Dict[str, int], qubits: Optional[List[Union[Qubit, int]]] = None):
        """Create a CircuitResult from a dictionary as returned by `qiskit.result.get_counts()`.
        The keys are little-endian bitstrings, i.e. the zeroth register is at the rightmost 
        position in the string.

        E.g., ``{"001": 5, "010": 4}`` to denote a result where the outcome with only the zeroth 
        qubit being 1 occured 5 times and the outcome of the first qubit begin 1 occured 5 times. 

        Optionally, the results on only a selection of qubits can be extracted (marginalization).  
        Example: 

        >>> CircuitResult({"101": 9, "001": 4}, [0, 2])

        will result in storing ``{"11": 9, "01": 4}``. 

        Parameters
        ----------
        counts : Dict[str, int]
            Outcome/count dictionary
        qubits : Optional[List[Qubit | int]], optional
            If specified, only the given qubits are extracted. 
        """
        self.results: List[BinaryResult] = []

        if qubits is None:
            if len(counts) != 0:
                self.num_qubits = len(next(iter(counts)).replace(" ", ""))
            for key, value in counts.items():
                self.results.append(BinaryResult(Bitstring(int(key.replace(" ", ""), 2)), value))

        else:
            if isinstance(qubits[0], Qubit):
                qubit_list: List[int] = [qubit.index for qubit in qubits]  # type: ignore
            else:
                qubit_list = qubits  # type: ignore
            self.num_qubits = len(qubit_list)
            for key, value in counts.items():
                key = key.replace(" ", "")  # might contain spaces to separate registers
                if qubits is not None:
                    key = "".join(key[index] for index in qubit_list)
                self.results.append(BinaryResult(Bitstring(int(key, 2)), value))

    def __str__(self) -> str:
        return "CircuitResult(" + ", ".join(result.__str__(self.num_qubits) for result in self.results) + ")"


class StabilizerMeasurementFitter:

    def __init__(self, result: Result, circuit: QuantumCircuit):

        self.result = result
        try:
            self.readout_info: ReadoutInfo = circuit.metadata["readout info"]
        except (TypeError, KeyError):
            raise ValueError("The passed circuit does not seem to be generated by stabilizer_measurement_circuit() and cannot be evaluated")

    def expectation_values(self, full_hilbert_space=True) -> Dict[Pauli, float]:
        """Compute the expectation values for each Pauli measurement that has
        been made in the stabilizer measurement. 

        Parameters
        ----------
        full_hilbert_space : bool, optional
            In case the measurement has only been performed on m < n qubits, 
            return full length-n Paulis if this parameter is ``True`` or 
            return length-m Paulis with entries only on the measured qubits
            elsewise. 

        Returns
        -------
        Dict[Pauli, float]
            Dictionary of measured Paulis and their expectation values. 
        """
        expectation_values: Dict[Pauli, float] = {}
        readoud_circuit = self.readout_info.circuit
        inverse_circuit = readoud_circuit.inverse()
        num_qubits = readoud_circuit.num_qubits

        qubits = self.readout_info.qubits
        circuit_result = CircuitResult(self.result.get_counts(), qubits)  # type: ignore

        identity = Pauli("I" * num_qubits)
        for i in range(1, 2**num_qubits):
            z_pauli = identity.copy()
            for j in range(num_qubits):
                if i & (1 << j):
                    z_pauli[j] = 'Z'

            pauli = z_pauli.evolve(readoud_circuit)
            pauli.phase = 0
            pauli_z = pauli.evolve(inverse_circuit)
            assert pauli_z.phase == 2 or pauli_z.phase == 0
            expectation_value = _compute_expectation_value(circuit_result, i)

            if pauli_z.phase == 2:
                expectation_value *= -1
            expectation_values[pauli] = expectation_value

        expectation_values[identity] = 1

        if qubits is None:
            return expectation_values

        # Only a subset of qubits were measured, we want to insert the missing
        # identities into the Paulis
        full_expectation_values = {}
        full_identity = Pauli("I" * self.readout_info.total_num_qubits)
        for key, value in expectation_values.items():
            qubit_list: List[int] = [qubit.index for qubit in qubits]
            new_key: Pauli = full_identity.copy()
            for index, qubit in enumerate(qubit_list):
                new_key[qubit] = key[index]
            full_expectation_values[new_key] = value

        return full_expectation_values

    def density_matrix(self, full_hilbert_space=True) -> np.ndarray:
        """Compute the density matrix that can be reconstructed
        from the measured Pauli operators. 

        Parameters
        ----------
        full_hilbert_space : bool, optional
            In case the measurement has only been performed on m < n qubits, 
            return a matrix in 2^n-dimensional Hilbern space if this parameter 
            is ``True`` or only 2^m-dimensional elsewise. 

        Returns
        -------
        np.ndarray
            Computed density matrix
        """
        return _compute_density_matrix_from_pauli_expectation_values(self.readout_info.total_num_qubits,
                                                                     self.expectation_values())


def _compute_expectation_value(circuit_result: CircuitResult, s: Bitstring):
    expectation_value: int = 0
    total_count: int = 0
    for result in circuit_result.results:
        if (s & result.bitstring).bit_count() & 1:
            expectation_value -= result.count
        else:
            expectation_value += result.count
        total_count += result.count
    return expectation_value / total_count


def _compute_density_matrix_from_pauli_expectation_values(num_qubits: int, expectation_values: Dict[Pauli, float]) -> np.ndarray:
    """Given a dictionary of expectation values, reconstruct the density matrix
    through linear combination

    Parameters
    ----------
    num_qubits : int
        Number of qubits
    expectation_values : Dict[Pauli, float]
        Dictionary of expectation values for Pauli measurements

    Returns
    -------
    np.ndarray
        Density matrix
    """
    density_matrix = np.zeros(shape=[2**num_qubits, 2**num_qubits], dtype=np.complex128)
    for pauli, expectation_value in expectation_values.items():
        density_matrix += pauli.to_matrix() * expectation_value
    density_matrix *= (1. / 2**num_qubits)
    return density_matrix
