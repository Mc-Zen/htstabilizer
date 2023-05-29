import numpy as np
from typing import List, Tuple, Union

from qiskit import QuantumCircuit
from .graph import Graph
from . import f2_algebra as f2


class Stabilizer:
    """Description of a stabilizer group for qubit systems."""

    def __init__(self, data: Union[List[str], Graph, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], QuantumCircuit], validate: bool = False):
        """Create an n-qubit stabilizer table from either
          - a list of n Pauli strings (need to commute and be independant) making up the generator
            e.g. ["XY", "-YZ", "+ZX"]. The first Pauli character corresponds to the first qubit. 
            The phase can be "+" or "-" and may be omitted if positive. Every character other than 
            +, -, X, Y, Z is treated as identity (I). 
          - a graph representing a graph state
          - or a pair of n×n matrices encoding the X and Z components of the generator, in the following way:
            Given a set of Paulis {P_1,...,P_n}, then the X and Z matrices are given by
                 R = | P_1[0].x   P_2[0].x   P_3[0].x |      S = | P_1[0].z   P_2[0].z   P_3[0].z |
                     | P_1[1].x   P_2[1].x   P_3[1].x |          | P_1[1].z   P_2[1].z   P_3[1].z |
                     |   .          .          .      |          |   .          .          .      |
                     | P_1[n].x   P_2[n].x   P_3[n].x |          | P_1[n].z   P_2[n].z   P_3[n].z |
            respectively where P_i[j].x denotes the x component (0 or 1) of the jth qubit and similar for z. 
            An optional third numpy array of length n may be passed to specify the phases for each Pauli
            respectively where 0 corresponds to positive phase and 1 to negative phase. 
          - A qiskit QuantumCircuit that has only Clifford gates as instructions. 

        Examples
        --------

        Below some examples that result in the exact same stabilizer. 
        ```
        s1 = Stabilizer(["XZZ", "ZXI", "ZIX"])

        s2 = Stabilizer(Graph.star(3))

        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        S = np.array([[0, 1, 1],
                      [1, 0, 0],
                      [1, 0, 0]])
        s3 = Stabilizer((R, S))
        s4 = Stabilizer((R, S, np.array([0,0,0])))
        ```


        Parameters
        ----------
        data : List[str] | Graph | Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray] | QuantumCircuit
            Input data in form of one of the versions specified above. 
        validate : bool, optional
            Whether to validate the stabilizer, i.e. check that all Paulis commute and 
            that the generator is linear independant, by default False
        """
        if isinstance(data, tuple) and (list(map(type, data)) == [np.ndarray, np.ndarray] or list(map(type, data)) == [np.ndarray, np.ndarray, np.ndarray]):
            ZX, ZZ = data[0], data[1]
            assert ZX.shape[0] == ZX.shape[1] == ZZ.shape[0] == ZZ.shape[1], "Input stabilizer matrices need to be square and of same dimensions"
            self.R = ZX
            self.S = ZZ
            self.num_qubits = ZX.shape[0]
            if len(data) == 3:
                self.phases = data[2].astype(np.int8)
            else:
                self.phases = np.zeros(self.num_qubits, dtype=np.int8)

            if self.R.dtype != np.int8:
                self.R = self.R.astype(np.int8)
            if self.S.dtype != np.int8:
                self.S = self.S.astype(np.int8)
        elif isinstance(data, list):
            self.num_qubits = len(data[0].lstrip("+-"))
            self.R = np.zeros((self.num_qubits, self.num_qubits), dtype=np.int8)
            self.S = np.zeros((self.num_qubits, self.num_qubits), dtype=np.int8)
            self.phases = np.zeros(self.num_qubits, dtype=np.int8)
            assert len(data) == self.num_qubits, f"Expected {self.num_qubits} Paulis for a {self.num_qubits}-qubit stabilizer"

            for row, pauli in enumerate(data):
                offset = 0
                if pauli.startswith("+"):
                    offset = 1
                elif pauli.startswith("-"):
                    offset = 1
                    self.phases[row] = 1
                if self.num_qubits != len(pauli) - offset:
                    raise ValueError("Not all paulis in the list have the same length")
                for col, character in enumerate(pauli[offset:]):
                    self.R[col, row] = int(character in "XY")
                    self.S[col, row] = int(character in "ZY")
        elif isinstance(data, Graph):
            self.num_qubits = data.num_vertices
            self.R = np.eye(self.num_qubits, dtype=np.int8)
            self.S = data.adjacency_matrix
            self.phases = np.zeros(self.num_qubits, dtype=np.int8)

            if self.R.dtype != np.int8:
                self.R = self.R.astype(np.int8)
            if self.S.dtype != np.int8:
                self.S = self.S.astype(np.int8)
        elif isinstance(data, QuantumCircuit):
            from qiskit.quantum_info import StabilizerState
            stabilizer_state = StabilizerState(data)
            self.num_qubits = stabilizer_state.num_qubits
            self.R = stabilizer_state.clifford.tableau[self.num_qubits: 2 * self.num_qubits, :self.num_qubits].astype(np.int8).T
            self.S = stabilizer_state.clifford.tableau[self.num_qubits: 2 * self.num_qubits, self.num_qubits:-1].astype(np.int8).T
            self.phases = stabilizer_state.clifford.tableau[self.num_qubits: 2 * self.num_qubits, -1].astype(np.int8).T
        else:
            assert False, "Unsupported input data"

        if validate:
            assert self.validate(), "Stabilizer verification failed: This stabilizer is not vaild"

    def validate(self) -> bool:
        """Verify that this is a valid stabilizer, i.e. that all Paulis commute and 
        that the generator is linear independant. 
        """
        n = self.num_qubits
        RS = np.concatenate([self.R, self.S])
        rank = f2.rank(RS)
        zero = np.zeros((n, n), dtype=np.int8)
        I = np.eye(n, dtype=np.int8)
        symp = np.block([[zero, I],
                         [I,    zero]])
        return rank == self.num_qubits and not np.any(f2.mat_mul(f2.mat_mul(RS.T, symp), RS))

    def expand(self) -> Tuple[np.ndarray, np.ndarray]:
        """Expand the n-element generator of the stabilizer group which is given by ZX and ZZ matrices
        to the full, 2^n-element stabilizer group that the generator spans.

        Warning: this operation is inherently inefficient and should only be used for small qubit numbers.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X and Z matrices of the shape (n, 2^n) respectively
        """
        n = self.num_qubits
        X = np.zeros((n, 1 << n), dtype=np.int8)
        Z = np.zeros((n, 1 << n), dtype=np.int8)

        for i in range(1 << n):
            for j in range(n):
                if i & (1 << j):
                    X[:, i] ^= self.R[:, j]
                    Z[:, i] ^= self.S[:, j]
        return X, Z

    def is_qubit_entangled(self, qubit: int) -> bool:
        """Determine if the given qubit is entangled in the stabilizer state
        described by the stabilizer.
        """
        qubit_pauli = 0
        for i in range(0, self.num_qubits):
            pauli = (self.R[qubit, i] << 1) | self.S[qubit, i]
            if pauli == 0:
                continue
            if qubit_pauli == 0:
                qubit_pauli = pauli
            else:
                if qubit_pauli != pauli:
                    return True
        return False

    def __eq__(self, other) -> bool:
        return self.num_qubits == other.num_qubits and \
            np.array_equal(self.R, other.R) and \
            np.array_equal(self.S, other.S) and \
            np.array_equal(self.phases, other.phases)

    def is_equivalent_mod_phase(self, other) -> bool:
        """Check whether both stabilizers span the same stabilizer group 
        up to phases.  

        Note, that this is different to `__eq__(self, other)` because the
        latter only checks if the generator is exactly the same one. Two
        different generators (in the simplest case just reordered ones)
        can still span the same stabilizer group. 

        This version ignores phases of the Paulis. See also
        `is_equivalent()`

        Parameters
        ----------
        other : Stabilizer
            Stabilizer to compare with

        Returns
        -------
        bool
            True, if the same group is spanned by both stabilizers up to phases
        """
        if self.num_qubits != other.num_qubits:
            return False
        n = self.num_qubits
        RS = np.concatenate([self.R, self.S])
        RS_other = np.concatenate([other.R, other.S])
        zero = np.zeros((n, n), dtype=np.int8)
        I = np.eye(n, dtype=np.int8)
        symp = np.block([[zero, I],
                         [I,    zero]])
        return not np.any(f2.mat_mul(f2.mat_mul(RS.T, symp), RS_other))

    def is_equivalent(self, other) -> bool:
        """Check whether both stabilizers span the same stabilizer group. 
        Note, that this is different to `__eq__(self, other)` because the
        latter only checks if the generator is exactly the same one. Two
        different generators (in the simplest case just reordered ones)
        can still span the same stabilizer group. 

        This version does NOT ignore phases of the Paulis. See also
        `is_equivalent_mod_phase()`

        Parameters
        ----------
        other : Stabilizer
            Stabilizer to compare with

        Returns
        -------
        bool
            True, if the same group is spanned by both stabilizers
        """
        n = self.num_qubits
        if n != other.num_qubits:
            return False
        m = np.block([[self.R.T,  self.S.T, self.phases.reshape((n, 1))],
                      [other.R.T, other.S.T, other.phases.reshape((n, 1))]])
        return f2.rank(m) == n

    def to_list(self) -> List[str]:
        """Get a list of Pauli strings, e.g. ["+XY", "-ZZ"]
        """
        chs = ["I", "X", "Z", "Y"]
        phs = ["+", "-"]
        return [phs[self.phases[j]] + "".join(chs[2*self.S[i][j] + self.R[i][j]] for i in range(self.num_qubits)) for j in range(self.num_qubits)]

    def __repr__(self) -> str:
        content = "','".join(self.to_list())
        return f"Stabilizer(['{content}'])"
