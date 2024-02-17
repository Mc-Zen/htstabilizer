from src.htstabilizer.graph import Graph
from src.htstabilizer import circuit_lookup
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
        for num_qubits in range(5, 6):
            for i in range(100):
                qc: QuantumCircuit = random_clifford(num_qubits).to_circuit()  # type: ignore
                stabilizer = Stabilizer(qc)

                x_gates = QuantumCircuit(num_qubits)
                for qubit in range(num_qubits):
                    if np.random.random() > .5:
                        x_gates.x(qubit)
                qc2: QuantumCircuit = x_gates.compose(qc)  # type: ignore

                qc3 = rotate_stabilizer_into_state(qc, Stabilizer(qc2))
                assert_same_state(qc2, qc3)
                qc3 = rotate_stabilizer_into_state(qc2, Stabilizer(qc))
                assert_same_state(qc, qc3)

    def get_graph_state_circuit(self, graph: Graph):
        graph_state_circuit = QuantumCircuit(graph.num_vertices)
        graph_state_circuit.h(range(graph.num_vertices))
        edges = graph.get_edges()
        for edge in edges:
            graph_state_circuit.cz(edge[0], edge[1])
        return graph_state_circuit

    def test_fail_case(self):
        qc = circuit_lookup.parse_circuit(5, "h0 h3 cz0,3 h1 cz0,1 s3 h3 h4 s4 cz3,4 h0 s0 h2 s2 h2 cz0,2 s0 h0 h3 s3 h3 cz0,3 s2 h3 ")
        print(qc)
        graph = Graph.decompress(5, 919)
        graph_state_circuit = self.get_graph_state_circuit(graph)
        qc_p = rotate_stabilizer_into_state(qc, Stabilizer(graph))
        print(qc_p)
        # qc_p = rotate_stabilizer_into_state(qc, graph_state_circuit)
        # print(qc_p)
        self.assertTrue(Stabilizer(graph_state_circuit).is_equivalent_mod_phase(Stabilizer(qc_p)))
        # self.assertTrue(Stabilizer(graph_state_circuit).is_equivalent(Stabilizer(qc_p)))
        assert_same_state(qc_p, graph_state_circuit)
