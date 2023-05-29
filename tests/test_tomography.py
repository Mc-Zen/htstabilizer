from src.htstabilizer.tomography import *
import qiskit
import qiskit.quantum_info as qi
import unittest
from numpy.testing import assert_allclose


def bell(n):
    assert n >= 2
    qc = QuantumCircuit(n)
    qc.h(range(2))
    qc.cx(0, 1)
    return qc


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


class TestStabilizerMeasurementCircuit(unittest.TestCase):

    def test_measure_all_qubits(self):
        qc_prep = bell(4)
        qc = stabilizer_measurement_circuit(qc_prep, Stabilizer(["ZIII", "IZII", "IIZI", "IIIZ"]))

        ref = QuantumCircuit(4)
        ref.compose(qc_prep, inplace=True)
        ref.h(range(4))
        ref.h(range(4))
        ref.measure_all()
        self.assertEqual(qc, ref)

    def test_measure_some_qubits(self):
        qc_prep = bell(4)
        qc = stabilizer_measurement_circuit(qc_prep, Stabilizer(["ZI", "IZ"]), measured_qubits=[2, 3])

        ref = QuantumCircuit(4)
        ref.compose(qc_prep, inplace=True)
        ref.h(range(2, 4))
        ref.h(range(2, 4))
        ref.measure_all()
        self.assertEqual(qc, ref)

        backend_sim = qiskit.Aer.get_backend("qasm_simulator")
        job_sim = qiskit.execute(qc, backend=backend_sim, shots=20000)

        smc = StabilizerMeasurementFitter(job_sim.result(), qc)
        self.assertEqual(smc.density_matrix(full_hilbert_space=True).shape, (16, 16))
        self.assertEqual(smc.density_matrix(full_hilbert_space=False).shape, (4, 4))


class TestFullStateTomography(unittest.TestCase):

    def check_tomography(self, circuit: QuantumCircuit, measured_qubits=None, atol=.006):
        fst = full_state_tomography_circuits(circuit, measured_qubits=measured_qubits)
        print(len(fst))

        backend_sim = qiskit.Aer.get_backend("qasm_simulator")
        job_sim = qiskit.execute(fst, backend=backend_sim, seed_simulator=100,shots=10000)

        fitter = FullStateTomographyFitter(job_sim.result(), fst)
        density_matrix = fitter.density_matrix()
        density_matrix_theo = qi.DensityMatrix(circuit)
        assert_allclose(density_matrix, density_matrix_theo, atol=atol)

    def test_measure_all_qubits_4(self):
        qc_prep = bell(4)
        fst = full_state_tomography_circuits(qc_prep)
        self.assertEqual(type(fst), list)
        self.assertEqual(type(fst[0]), QuantumCircuit)

    def test_bell(self):
        qc_prep = bell(4)
        self.check_tomography(qc_prep, atol=.0054)

    def test_1pp(self):
        qc_prep = QuantumCircuit(3)
        qc_prep.h([1, 2])
        self.check_tomography(qc_prep, atol=.0089)

    def test_uuu(self):
        qc_prep = QuantumCircuit(3)
        qc_prep.u(1.2342, -0.2343, 0., 0)
        qc_prep.u(-1.542, -0.7461, 0., 1)
        qc_prep.u(0.0156, 0.98324, 0., 2)
        self.check_tomography(qc_prep, atol=.0064)

    def test_uuuu(self):
        qc_prep = QuantumCircuit(4)
        qc_prep.u(1.2342, -0.2343, 0., 0)
        qc_prep.u(-1.542, -0.7461, 0., 1)
        qc_prep.u(-2.222, 0.98610, 0., 2)
        qc_prep.u(0.0156, 0.98324, 0., 3)
        self.check_tomography(qc_prep, atol=.0053)

    def test_ghz5(self):
        qc_prep = QuantumCircuit(5)
        qc_prep.h(0)
        qc_prep.cx(0, range(1,5))
        self.check_tomography(qc_prep, atol=.0050)

    def test_pauli_from_bitstring(self):
        self.assertEqual(z_pauli_from_bitstring(1, 0b0), Pauli("I"))
        self.assertEqual(z_pauli_from_bitstring(3, 0b000), Pauli("III"))
        self.assertEqual(z_pauli_from_bitstring(3, 0b011), Pauli("IZZ"))
        self.assertEqual(z_pauli_from_bitstring(4, 0b1011), Pauli("ZIZZ"))