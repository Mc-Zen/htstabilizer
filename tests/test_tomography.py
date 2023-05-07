from src.htstabilizer.tomography import *
import qiskit
import qiskit.quantum_info as qi
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


class TestFullStateTomography(unittest.TestCase):

    def test_measure_all_qubits_4(self):
        qc_prep = QuantumCircuit(4)
        qc_prep.h(range(2))
        qc_prep.cx(0, 1)
        fst = full_state_tomography_circuits(qc_prep)
        self.assertEqual(type(fst), list)
        self.assertEqual(type(fst[0]), QuantumCircuit)

        backend_sim = qiskit.Aer.get_backend("qasm_simulator")
        job_sim = qiskit.execute(fst, backend=backend_sim, shots=100)
        result = job_sim.result()
        print(result)

        fitter = FullStateTomographyFitter(result, fst)
        print(fitter.expectation_values())
        print(fitter.density_matrix())

    def test_measure_all_qubits_2(self):
        qc_prep = QuantumCircuit(2)
        # qc_prep.h(0)
        qc_prep.h(range(2))
        qc_prep.s(1)
        qc_prep.cx(0, 1)
        fst = full_state_tomography_circuits(qc_prep)

        backend_sim = qiskit.Aer.get_backend("qasm_simulator")
        job_sim = qiskit.execute(fst, backend=backend_sim, shots=10000)
        result = job_sim.result()
        print(result)

        fitter = FullStateTomographyFitter(result, fst)
        print(fitter.expectation_values())
        
        with np.printoptions(precision=2, suppress=True):
            print(fitter.density_matrix())
        print(qi.DensityMatrix(qc_prep))

    def test_measure_by_register(self):
        qr = QuantumRegister(4)
        qc_prep = QuantumCircuit(qr)
        qc_prep.h(range(2))
        qc_prep.cx(0, 1)
        qc = full_state_tomography_circuits(qc_prep, measured_qubits=qr)

    def test_measure_some_by_index(self):
        qc_prep = QuantumCircuit(4)
        qc_prep.h(range(2))
        qc_prep.cx(0, 1)
        qc = full_state_tomography_circuits(qc_prep, measured_qubits=[2, 3])

    def test_measure_some_by_qubit(self):
        qr = QuantumRegister(4)
        qc_prep = QuantumCircuit(qr)
        qc_prep.h(range(2))
        qc_prep.cx(0, 1)
        qc = full_state_tomography_circuits(qc_prep, measured_qubits=[qr[2], qr[3]])


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
