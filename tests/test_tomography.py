from src.htstabilizer.tomography import *

import unittest


class TestStabilizerMeasurementCircuit(unittest.TestCase):

    def test_measure_all_qubits(self):
        qc_prep = QuantumCircuit(4)
        qc_prep.h(range(2))
        qc_prep.cx(0, 1)
        qc = stabilizer_measurement_circuit(qc_prep, Stabilizer(["ZIII", "IZII", "IIZI", "IIIZ"]))
        self.assertEqual(type(qc), QuantumCircuit)

        ref = QuantumCircuit(4)
        ref.compose(qc_prep, inplace=True)
        ref.h(range(4))
        ref.h(range(4))
        ref.measure_all()
        self.assertEqual(qc, ref)

    def test_measure_by_register(self):
        qr = QuantumRegister(4)
        qc_prep = QuantumCircuit(qr)
        qc_prep.h(range(2))
        qc_prep.cx(0, 1)
        qc = stabilizer_measurement_circuit(qc_prep, Stabilizer(["ZIII", "IZII", "IIZI", "IIIZ"]), measured_qubits=qr)
        self.assertEqual(type(qc), QuantumCircuit)

        ref = QuantumCircuit(4)
        ref.compose(qc_prep, inplace=True)
        ref.h(range(2, 4))
        ref.h(range(2, 4))
        ref.measure_all()

    def test_measure_some_by_index(self):
        qc_prep = QuantumCircuit(4)
        qc_prep.h(range(2))
        qc_prep.cx(0, 1)
        qc = stabilizer_measurement_circuit(qc_prep, Stabilizer(["ZI", "IZ"]), measured_qubits=[2, 3])
        self.assertEqual(type(qc), QuantumCircuit)

        ref = QuantumCircuit(4)
        ref.compose(qc_prep, inplace=True)
        ref.h(range(2, 4))
        ref.h(range(2, 4))
        ref.measure_all()

    def test_measure_some_by_qubit(self):
        qr = QuantumRegister(4)
        qc_prep = QuantumCircuit(qr)
        qc_prep.h(range(2))
        qc_prep.cx(0, 1)
        qc = stabilizer_measurement_circuit(qc_prep, Stabilizer(["ZI", "IZ"]), measured_qubits=[qr[2], qr[3]])
        self.assertEqual(type(qc), QuantumCircuit)

        ref = QuantumCircuit(4)
        ref.compose(qc_prep, inplace=True)
        ref.h(range(2, 4))
        ref.h(range(2, 4))
        ref.measure_all()


class TestCircuitResult(unittest.TestCase):

    def test_(self):
        cr = CircuitResult({"110": 3, "001": 4})
        self.assertEqual(cr.results, [BinaryResult(6, 3), BinaryResult(1, 4)])
        self.assertEqual(cr.num_qubits, 3)

    def test_1(self):
        cr = CircuitResult({"110": 3, "001": 4}, [0, 1, 2])
        self.assertEqual(cr.results, [BinaryResult(6, 3), BinaryResult(1, 4)])
        self.assertEqual(cr.num_qubits, 3)

    def test_selected_qubits(self):
        cr = CircuitResult({"101": 9, "001": 4}, [0, 2])
        self.assertEqual(cr.results, [BinaryResult(3, 9), BinaryResult(1, 4)])
        self.assertEqual(cr.num_qubits, 2)
