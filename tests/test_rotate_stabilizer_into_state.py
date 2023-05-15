from src.htstabilizer.stabilizer_circuits import _get_preparation_circuit_modulo_phase
from src.htstabilizer.rotate_stabilizer_into_state import *
from src.htstabilizer.stabilizer import Stabilizer
from qiskit.quantum_info import random_clifford

import unittest


class TestDoPrepareSameState(unittest.TestCase):

    def test_same_state(self):
        qc = QuantumCircuit(3)
        qc.s([1, 2])
        qc.h(1)
        qc.cx(1, 2)
        assert_same_state(qc, qc)

    def test_not_same_state(self):
        qc = QuantumCircuit(3)
        qc.s([1, 2])
        qc.h(1)
        qc.cx(1, 2)
        qc1 = qc.copy()
        qc.h(0)
        self.assertRaises(AssertionError, assert_same_state, qc, qc1)


class TestRotateStabilizerIntoState(unittest.TestCase):

    def get_different_bell_circuits(self):
        qc = QuantumCircuit(2)
        qc.h(range(2))
        qc.cz(0, 1)

        qc2 = qc.copy()
        qc2.s(1)
        qc2.h(0)
        qc2.s(0)
        qc2.h(0)
        return qc, qc2

    def test_basic(self):
        qc1, qc2 = self.get_different_bell_circuits()
        self.assertTrue(Stabilizer(qc2).is_equivalent_mod_phase(Stabilizer(qc1)))
        self.assertFalse(do_prepare_same_state(qc1, qc2))

        qc3 = rotate_stabilizer_into_state(qc1, qc2)
        assert_same_state(qc2, qc3)

    def test_basic_inplace(self):
        qc1, qc2 = self.get_different_bell_circuits()
        self.assertTrue(Stabilizer(qc2).is_equivalent_mod_phase(Stabilizer(qc1)))
        self.assertFalse(do_prepare_same_state(qc1, qc2))

        qc3 = rotate_stabilizer_into_state(qc1, qc2, inplace=True)
        self.assertTrue(qc3 is qc1)
        assert_same_state(qc1, qc2)

    def test_basic_2(self):
        qc1, qc2 = self.get_different_bell_circuits()
        qc3 = rotate_stabilizer_into_state(qc1, Stabilizer(qc2))
        assert_same_state(qc2, qc3)

        qc1, qc2 = qc2, qc1
        qc3 = rotate_stabilizer_into_state(qc1, Stabilizer(qc2))
        assert_same_state(qc2, qc3)

    def test_special_case(self):
        stabilizer = Stabilizer(['+YX', '+ZY'])
        qc = _get_preparation_circuit_modulo_phase(stabilizer, "all")
        qc2 = rotate_stabilizer_into_state(qc, stabilizer)
        self.assertTrue(stabilizer.is_equivalent(Stabilizer(qc2)))

    def test_basic_inplace_2(self):
        qc1, qc2 = self.get_different_bell_circuits()
        qc3 = rotate_stabilizer_into_state(qc1, Stabilizer(qc2), inplace=True)
        assert_same_state(qc1, qc3)

        qc1, qc2 = qc2, qc1
        qc3 = rotate_stabilizer_into_state(qc1, Stabilizer(qc2), inplace=True)
        assert_same_state(qc1, qc3)

    def test_random_2(self):
        for num_qubits in range(1, 6):
            for i in range(100):
                qc: QuantumCircuit = random_clifford(num_qubits).to_circuit()  # type: ignore
                stabilizer = Stabilizer(qc)

                x_gates = QuantumCircuit(num_qubits)
                for qubit in range(num_qubits):
                    if np.random.random() > .5:
                        x_gates.x(qubit)
                qc2: QuantumCircuit = x_gates.compose(qc) # type: ignore

                qc3 = rotate_stabilizer_into_state(qc, Stabilizer(qc2))
                assert_same_state(qc2, qc3)
                qc3 = rotate_stabilizer_into_state(qc2, Stabilizer(qc))
                assert_same_state(qc, qc3)
