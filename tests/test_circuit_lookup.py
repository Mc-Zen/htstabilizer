import unittest

from src.htcircuits.circuit_lookup import *
from src.htcircuits.lc_classes import *
from src.htcircuits.graph import Graph
from qiskit import transpile
from qiskit.circuit.library import GraphState
from qiskit.quantum_info import Statevector, StabilizerState
from numpy.testing import assert_almost_equal


class TestCircuitLookup(unittest.TestCase):

    def test_circuit_info(self):
        info = CircuitInfo(4, "23:4:3:cx0,1 cx2,3 cx3,2 cx2,1 h0")
        self.assertEqual(info.num_qubits, 4)
        self.assertEqual(info.graph_id, 23)
        self.assertEqual(info.cost, 4)
        self.assertEqual(info.depth, 3)
        self.assertEqual(info.circuit_string, "cx0,1 cx2,3 cx3,2 cx2,1 h0")
        circuit = info.parse_circuit()
        self.verify_cost_and_depth(info)

    def are_circuits_equivalent(self, c1, c2):
        statevector1 = np.array(Statevector(c1))
        statevector2 = np.array(Statevector(c2))
        ratio = statevector1[0] / statevector2[0]
        assert_almost_equal(statevector1, statevector2*ratio)
        # self.assertListEqual(statevector1, statevector2*ratio)
        # return self.assertEqual(statevector1, statevector2*ratio)

    def test_verify_cost_and_depth(self):
        self.verify_cost_and_depth_for_all(2, "all")

        self.verify_cost_and_depth_for_all(3, "all")
        self.verify_cost_and_depth_for_all(3, "linear")

        self.verify_cost_and_depth_for_all(4, "all")
        self.verify_cost_and_depth_for_all(4, "linear")
        self.verify_cost_and_depth_for_all(4, "star")
        self.verify_cost_and_depth_for_all(4, "cycle")

        self.verify_cost_and_depth_for_all(5, "all")
        self.verify_cost_and_depth_for_all(5, "linear")
        self.verify_cost_and_depth_for_all(5, "Q")
        self.verify_cost_and_depth_for_all(5, "T")
        self.verify_cost_and_depth_for_all(5, "star")
        self.verify_cost_and_depth_for_all(5, "cycle")

    def test_verify_state_2_all(self):
        self.verify_state_for_all(2, "all")

    def test_verify_state_3_all(self):
        self.verify_state_for_all(3, "all")

    def test_verify_state_3_linear(self):
        self.verify_state_for_all(3, "linear")

    def test_verify_state_4_all(self):
        self.verify_state_for_all(4, "all")

    def test_verify_state_5_all(self):
        self.verify_state_for_all(5, "all")

    def test_verify_state_5_linear(self):
        self.verify_state_for_all(5, "linear")

    def test_verify_stabilizer_2_all(self):
        self.verify_stabilizer_for_all(2, "all")

    def test_verify_stabilizer_3_all(self):
        self.verify_stabilizer_for_all(3, "all")

    def test_verify_stabilizer_3_linear(self):
        self.verify_stabilizer_for_all(3, "linear")

    def test_verify_stabilizer_4_all(self):
        self.verify_stabilizer_for_all(4, "all")

    def test_verify_stabilizer_4_linear(self):
        self.verify_stabilizer_for_all(4, "linear")

    def test_verify_stabilizer_4_star(self):
        self.verify_stabilizer_for_all(4, "star")

    def test_verify_stabilizer_4_cycle(self):
        self.verify_stabilizer_for_all(4, "cycle")

    def test_verify_stabilizer_5_all(self):
        self.verify_stabilizer_for_all(5, "all")

    def test_verify_stabilizer_5_linear(self):
        self.verify_stabilizer_for_all(5, "linear")

    def test_verify_stabilizer_5_star(self):
        self.verify_stabilizer_for_all(5, "star")

    def test_verify_stabilizer_5_cycle(self):
        self.verify_stabilizer_for_all(5, "cycle")

    def test_verify_stabilizer_5_T(self):
        self.verify_stabilizer_for_all(5, "T")

    def test_verify_stabilizer_5_Q(self):
        self.verify_stabilizer_for_all(5, "Q")

    ###

    def test_verify_lc_class_2_all(self):
        self.verify_lc_class_for_all(2, "all")

    def test_verify_lc_class_3_all(self):
        self.verify_lc_class_for_all(3, "all")

    def test_verify_lc_class_3_linear(self):
        self.verify_lc_class_for_all(3, "linear")

    def test_verify_lc_class_4_all(self):
        self.verify_lc_class_for_all(4, "all")

    def test_verify_lc_class_4_linear(self):
        self.verify_lc_class_for_all(4, "linear")

    def test_verify_lc_class_4_star(self):
        self.verify_lc_class_for_all(4, "star")

    def test_verify_lc_class_4_cycle(self):
        self.verify_lc_class_for_all(4, "cycle")

    def test_verify_lc_class_5_all(self):
        self.verify_lc_class_for_all(5, "all")

    def test_verify_lc_class_5_linear(self):
        self.verify_lc_class_for_all(5, "linear")

    def test_verify_lc_class_5_star(self):
        self.verify_lc_class_for_all(5, "star")

    def test_verify_lc_class_5_cycle(self):
        self.verify_lc_class_for_all(5, "cycle")

    def test_verify_lc_class_5_Q(self):
        self.verify_lc_class_for_all(5, "Q")

    def test_verify_lc_class_5_T(self):
        self.verify_lc_class_for_all(5, "T")

    def verify_cost_and_depth(self, info: CircuitInfo):
        circuit = info.parse_circuit()
        transpiled_circuit: QuantumCircuit = transpile(circuit, basis_gates=["cx", "h", "s"])
        self.assertEqual(transpiled_circuit.count_ops().get("cx", 0), info.cost)
        self.assertEqual(transpiled_circuit.depth(lambda instr: instr.operation.name == "cx"), info.depth)

    def verify_cost_and_depth_for_all(self, num_qubits, connectivity):
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5]
        cls = LCClasses[num_qubits - 2]
        for i in range(cls.count()):
            info = circuit_lookup(num_qubits, connectivity, i)
            self.verify_cost_and_depth(info)

    def verify_state(self, info: CircuitInfo):
        circuit = info.parse_circuit()
        # graph_state_circuit = GraphState(Graph.decompress(info.num_qubits, info.graph_id).adjacency_matrix)
        graph_state_circuit = QuantumCircuit(info.num_qubits)
        graph_state_circuit.h(range(info.num_qubits))
        edges = Graph.decompress(info.num_qubits, info.graph_id).get_edges()
        for edge in edges:
            graph_state_circuit.cz(edge[0], edge[1])
        # print(circuit.draw("text"))
        # print(graph_state_circuit.draw("text"))
        # print(Statevector(circuit))
        # print(Statevector(graph_state_circuit))
        # print(info.graph_id)
        self.are_circuits_equivalent(circuit, graph_state_circuit)
        # self.assertEqual(Statevector(circuit), Statevector(graph_state_circuit))

    def verify_state_for_all(self, num_qubits, connectivity):
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5]
        cls = LCClasses[num_qubits - 2]
        for id in range(cls.count()):
            info = circuit_lookup(num_qubits, connectivity, id)
            self.verify_state(info)

    def verify_stabilizer_for_all(self, num_qubits, connectivity):
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5]
        cls = LCClasses[num_qubits - 2]
        for id in range(cls.count()):
            info = circuit_lookup(num_qubits, connectivity, id)
            circuit_stabilizer = Stabilizer(info.parse_circuit())
            graph_state_stabilizer = Stabilizer(Graph.decompress(num_qubits, info.graph_id))
            # print(circuit_stabilizer, graph_state_stabilizer)
            self.assertTrue(graph_state_stabilizer.is_equivalent(circuit_stabilizer))

    def verify_lc_class_for_all(self, num_qubits, connectivity):
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5]
        cls = LCClasses[num_qubits - 2]
        for id in range(cls.count()):
            info = circuit_lookup(num_qubits, connectivity, id)
            lc_class = determine_lc_class(Stabilizer(Graph.decompress(num_qubits, info.graph_id)))
            self.assertEqual(id, lc_class.id())
