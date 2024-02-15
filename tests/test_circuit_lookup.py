import unittest
from src.htstabilizer.find_local_clifford_layer import find_local_clifford_layer, local_clifford_layer_to_circuit
from src.htstabilizer.stabilizer_circuits import get_preparation_circuit
from src.htstabilizer.rotate_stabilizer_into_state import rotate_stabilizer_into_state

from src.htstabilizer.circuit_lookup import *
from src.htstabilizer.connectivity_support import get_connectivity_graph, get_available_connectivities
from src.htstabilizer.lc_classes import *
from src.htstabilizer.graph import Graph
from qiskit import transpile
from qiskit.quantum_info import Statevector
from numpy.testing import assert_almost_equal


def respects_connectivity(circuit: QuantumCircuit, connectivity_graph) -> bool:
    """Check that all lookup circuits indeed respect the connectivity constraints that they promise"""
    two_qubit_gates = circuit.get_instructions("cx") + circuit.get_instructions("cz") + circuit.get_instructions("swap")
    for gate in two_qubit_gates:
        qubit1 = gate.qubits[0].index
        qubit2 = gate.qubits[1].index
        assert type(qubit1) is int and type(qubit2) is int
        if not connectivity_graph.has_edge(qubit1, qubit2):
            return False
    return True


class TestStabilizerCircuitLookupBase(unittest.TestCase):

    def verify_connectivity_for_all(self, num_qubits, connectivity):
        """Check that all lookup circuits indeed respect the connectivity constraints that they promise"""
        con = get_connectivity_graph(num_qubits, connectivity)
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5, LCClass6]
        cls = LCClasses[num_qubits - 2]
        for i in range(cls.count()):
            info = stabilizer_circuit_lookup(num_qubits, connectivity, i)
            circuit = info.parse_circuit()
            self.assertTrue(respects_connectivity(circuit, con))

    def verify_cost_and_depth(self, info: StabilizerCircuitInfo):
        """Verify that the lookup circuits cost and depth information is correct (compared with the given circuit)"""
        circuit = info.parse_circuit()
        transpiled_circuit: QuantumCircuit = transpile(circuit, basis_gates=["cx", "h", "s"])
        self.assertEqual(transpiled_circuit.count_ops().get("cx", 0), info.cost)
        self.assertEqual(transpiled_circuit.depth(lambda instr: instr.operation.name == "cx"), info.depth)

    def verify_cost_and_depth_for_all(self, num_qubits, connectivity):
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5, LCClass6]
        cls = LCClasses[num_qubits - 2]
        for i in range(cls.count()):
            info = stabilizer_circuit_lookup(num_qubits, connectivity, i)
            self.verify_cost_and_depth(info)

    def verify_state(self, info: StabilizerCircuitInfo):
        circuit = info.parse_circuit()
        graph = Graph.decompress(info.num_qubits, info.graph_id)
        # graph_state_circuit = GraphState(Graph.decompress(info.num_qubits, info.graph_id).adjacency_matrix)
        graph_state_circuit = QuantumCircuit(info.num_qubits)
        graph_state_circuit.h(range(info.num_qubits))
        edges = graph.get_edges()
        for edge in edges:
            graph_state_circuit.cz(edge[0], edge[1])
        # rotate_stabilizer_into_state(circuit, Stabilizer(graph), inplace=True)
        rotate_stabilizer_into_state(circuit, graph_state_circuit, inplace=True)
        # print(circuit.draw("text"))
        # print(graph_state_circuit.draw("text"))
        # print(Statevector(circuit))
        # print(Statevector(graph_state_circuit))
        # print(info.graph_id)
        self.are_circuits_equivalent(circuit, graph_state_circuit)
        # self.assertEqual(Statevector(circuit), Statevector(graph_state_circuit))

    def verify_state_for_all(self, num_qubits, connectivity):
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5, LCClass6]
        cls = LCClasses[num_qubits - 2]
        for id in range(cls.count()):
            with self.subTest(id=id):
                info = stabilizer_circuit_lookup(num_qubits, connectivity, id)
                self.verify_state(info)

    def verify_stabilizer_for_all(self, num_qubits, connectivity):
        """Check that the circuits have indeed the same stabilizers as the graphs states that they promise to generate"""
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5, LCClass6]
        cls = LCClasses[num_qubits - 2]
        for id in range(cls.count()):
            info = stabilizer_circuit_lookup(num_qubits, connectivity, id)
            circuit_stabilizer = Stabilizer(info.parse_circuit())
            graph_state_stabilizer = Stabilizer(Graph.decompress(num_qubits, info.graph_id))
            # print(circuit_stabilizer, graph_state_stabilizer)
            self.assertTrue(graph_state_stabilizer.is_equivalent_mod_phase(circuit_stabilizer))

    def verify_lc_class_for_all(self, num_qubits, connectivity):
        """Check that the graph info for each lookup entry coincides with the lc class it should have"""
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5, LCClass6]
        cls = LCClasses[num_qubits - 2]
        for id in range(cls.count()):
            info = stabilizer_circuit_lookup(num_qubits, connectivity, id)
            lc_class = determine_lc_class(Stabilizer(Graph.decompress(num_qubits, info.graph_id)))
            self.assertEqual(id, lc_class.id())

    def are_circuits_equivalent(self, c1, c2):
        from src.htstabilizer.rotate_stabilizer_into_state import assert_same_state
        assert_same_state(c1, c2)
        return
        statevector1 = np.array(Statevector(c1))
        statevector2 = np.array(Statevector(c2))
        ratio = statevector1[0] / statevector2[0]
        assert_almost_equal(statevector1, statevector2*ratio)
        # self.assertListEqual(statevector1, statevector2*ratio)
        # return self.assertEqual(statevector1, statevector2*ratio)


class TestStabilizerCircuitLookup_2_all(TestStabilizerCircuitLookupBase):
    def test_verify_state_2_all(self):
        self.verify_state_for_all(2, "all")

    def test_verify_stabilizer_2_all(self):
        self.verify_stabilizer_for_all(2, "all")

    def test_verify_lc_class_2_all(self):
        self.verify_lc_class_for_all(2, "all")

    def test_verify_connectivity_2_all(self):
        self.verify_connectivity_for_all(2, "all")


class TestStabilizerCircuitLookup_3_all(TestStabilizerCircuitLookupBase):
    def test_verify_state_3_all(self):
        self.verify_state_for_all(3, "all")

    def test_verify_stabilizer_3_all(self):
        self.verify_stabilizer_for_all(3, "all")

    def test_verify_lc_class_3_all(self):
        self.verify_lc_class_for_all(3, "all")

    def test_verify_connectivity_3_all(self):
        self.verify_connectivity_for_all(3, "all")


class TestStabilizerCircuitLookup_3_linear(TestStabilizerCircuitLookupBase):
    def test_verify_state_3_linear(self):
        self.verify_state_for_all(3, "linear")

    def test_verify_stabilizer_3_linear(self):
        self.verify_stabilizer_for_all(3, "linear")

    def test_verify_lc_class_3_linear(self):
        self.verify_lc_class_for_all(3, "linear")

    def test_verify_connectivity_3_linear(self):
        self.verify_connectivity_for_all(3, "linear")


class TestStabilizerCircuitLookup_4_all(TestStabilizerCircuitLookupBase):
    def test_verify_state_4_all(self):
        self.verify_state_for_all(4, "all")

    def test_verify_stabilizer_4_all(self):
        self.verify_stabilizer_for_all(4, "all")

    def test_verify_lc_class_4_all(self):
        self.verify_lc_class_for_all(4, "all")

    def test_verify_connectivity_4_all(self):
        self.verify_connectivity_for_all(4, "all")


class TestStabilizerCircuitLookup_4_linear(TestStabilizerCircuitLookupBase):

    def test_verify_state_4_linear(self):
        self.verify_state_for_all(4, "linear")

    def test_verify_stabilizer_4_linear(self):
        self.verify_stabilizer_for_all(4, "linear")

    def test_verify_lc_class_4_linear(self):
        self.verify_lc_class_for_all(4, "linear")

    def test_verify_connectivity_4_linear(self):
        self.verify_connectivity_for_all(4, "linear")


class TestStabilizerCircuitLookup_4_star(TestStabilizerCircuitLookupBase):
    def test_verify_state_4_star(self):
        self.verify_state_for_all(4, "star")

    def test_verify_stabilizer_4_star(self):
        self.verify_stabilizer_for_all(4, "star")

    def test_verify_lc_class_4_star(self):
        self.verify_lc_class_for_all(4, "star")

    def test_verify_connectivity_4_star(self):
        self.verify_connectivity_for_all(4, "star")


class TestStabilizerCircuitLookup_4_cycle(TestStabilizerCircuitLookupBase):
    def test_verify_state_4_cycle(self):
        self.verify_state_for_all(4, "cycle")

    def test_verify_stabilizer_4_cycle(self):
        self.verify_stabilizer_for_all(4, "cycle")

    def test_verify_lc_class_4_cycle(self):
        self.verify_lc_class_for_all(4, "cycle")

    def test_verify_connectivity_4_cycle(self):
        self.verify_connectivity_for_all(4, "cycle")


class TestStabilizerCircuitLookup_5_all(TestStabilizerCircuitLookupBase):
    def test_verify_state_5_all(self):
        self.verify_state_for_all(5, "all")

    def test_verify_stabilizer_5_all(self):
        self.verify_stabilizer_for_all(5, "all")

    def test_verify_lc_class_5_all(self):
        self.verify_lc_class_for_all(5, "all")

    def test_verify_connectivity_5_all(self):
        self.verify_connectivity_for_all(5, "all")


class TestStabilizerCircuitLookup_5_linear(TestStabilizerCircuitLookupBase):
    def test_verify_state_5_linear(self):
        self.verify_state_for_all(5, "linear")

    def test_verify_stabilizer_5_linear(self):
        self.verify_stabilizer_for_all(5, "linear")

    def test_verify_lc_class_5_linear(self):
        self.verify_lc_class_for_all(5, "linear")

    def test_verify_connectivity_5_linear(self):
        self.verify_connectivity_for_all(5, "linear")


class TestStabilizerCircuitLookup_5_star(TestStabilizerCircuitLookupBase):

    def test_verify_state_5_star(self):
        self.verify_state_for_all(5, "star")

    def test_verify_stabilizer_5_star(self):
        self.verify_stabilizer_for_all(5, "star")

    def test_verify_lc_class_5_star(self):
        self.verify_lc_class_for_all(5, "star")

    def test_verify_connectivity_5_star(self):
        self.verify_connectivity_for_all(5, "star")


class TestStabilizerCircuitLookup_5_cycle(TestStabilizerCircuitLookupBase):
    def test_verify_state_5_linear(self):
        self.verify_state_for_all(5, "cycle")

    def test_verify_stabilizer_5_cycle(self):
        self.verify_stabilizer_for_all(5, "cycle")

    def test_verify_lc_class_5_cycle(self):
        self.verify_lc_class_for_all(5, "cycle")

    def test_verify_connectivity_5_cycle(self):
        self.verify_connectivity_for_all(5, "cycle")


class TestStabilizerCircuitLookup_5_T(TestStabilizerCircuitLookupBase):

    def test_verify_state_5_T(self):
        self.verify_state_for_all(5, "T")

    def test_verify_stabilizer_5_T(self):
        self.verify_stabilizer_for_all(5, "T")

    def test_verify_lc_class_5_T(self):
        self.verify_lc_class_for_all(5, "T")

    def test_verify_connectivity_5_T(self):
        self.verify_connectivity_for_all(5, "T")


class TestStabilizerCircuitLookup_5_Q(TestStabilizerCircuitLookupBase):
    def test_verify_state_5_Q(self):
        self.verify_state_for_all(5, "Q")

    def test_verify_stabilizer_5_Q(self):
        self.verify_stabilizer_for_all(5, "Q")

    def test_verify_lc_class_5_Q(self):
        self.verify_lc_class_for_all(5, "Q")

    def test_verify_connectivity_5_Q(self):
        self.verify_connectivity_for_all(5, "Q")


class TestStabilizerCircuitLookup_6_all(TestStabilizerCircuitLookupBase):
    def test_verify_state_6_all(self):
        self.verify_state_for_all(6, "all")

    def test_verify_stabilizer_6_all(self):
        self.verify_stabilizer_for_all(6, "all")

    def test_verify_lc_class_6_all(self):
        self.verify_lc_class_for_all(6, "all")

    def test_verify_connectivity_6_all(self):
        self.verify_connectivity_for_all(6, "all")


class TestStabilizerCircuitLookup_6_linear(TestStabilizerCircuitLookupBase):
    def test_verify_stabilizer_6_linear(self):
        self.verify_stabilizer_for_all(6, "linear")

    def test_verify_lc_class_6_linear(self):
        self.verify_lc_class_for_all(6, "linear")

    def test_verify_state_6_linear(self):
        self.verify_state_for_all(6, "linear")

    def test_verify_connectivity_6_linear(self):
        self.verify_connectivity_for_all(6, "linear")


class TestStabilizerCircuitLookup_6_linear(TestStabilizerCircuitLookupBase):
    def test_verify_stabilizer_6_linear(self):
        self.verify_stabilizer_for_all(6, "linear")

    def test_verify_lc_class_6_linear(self):
        self.verify_lc_class_for_all(6, "linear")

    def test_verify_state_6_linear(self):
        self.verify_state_for_all(6, "linear")

    def test_verify_connectivity_6_linear(self):
        self.verify_connectivity_for_all(6, "linear")


class TestStabilizerCircuitLookup_6_star(TestStabilizerCircuitLookupBase):
    def test_verify_stabilizer_6_star(self):
        self.verify_stabilizer_for_all(6, "star")

    def test_verify_stabilizer_6_ame_state(self):
        info = stabilizer_circuit_lookup(6, "allx", 759)
        graph = Graph.decompress(info.num_qubits, info.graph_id)

        stabilizer = Stabilizer(graph)

        layer = find_local_clifford_layer(stabilizer.R, stabilizer.S, graph)
        if layer is None:
            raise RuntimeError("No circuit could be found. Please validate the input stabilizer.")
        layer_circuit = local_clifford_layer_to_circuit(layer).inverse()
        circuit = info.parse_circuit()  # .compose(layer_circuit)  # type: ignore

        graph_state_circuit = QuantumCircuit(info.num_qubits)
        graph_state_circuit.h(range(info.num_qubits))
        edges = graph.get_edges()
        for edge in edges:
            graph_state_circuit.cz(edge[0], edge[1])
        # print(circuit.draw("text"))
        # print(graph_state_circuit.draw("text"))
        # print(Statevector(circuit))
        # print(Statevector(graph_state_circuit))
        # print(info.graph_id)
        rotate_stabilizer_into_state(circuit, graph_state_circuit, inplace=True)

        def sv(c):
            arr = np.array(Statevector(c))
            # with np.printoptions(precision=0):
            print(np.real(arr/arr[0]))

        print()
        print()
        sv(circuit)
        sv(graph_state_circuit)
        print(graph_state_circuit)
        print(rotate_stabilizer_into_state(info.parse_circuit(), stabilizer, inplace=False))
        print(circuit)
        # circuit = get_preparation_circuit(Stabilizer(graph), "star")
        self.are_circuits_equivalent(circuit, graph_state_circuit)
    # def test_verify_lc_class_6_star(self):
    #     self.verify_lc_class_for_all(6, "star")

    def test_verify_lc_class_6_star(self):
        self.verify_lc_class_for_all(6, "star")

    def test_verify_state_6_star(self):
        self.verify_state_for_all(6, "star")

    def test_verify_connectivity_6_star(self):
        self.verify_connectivity_for_all(6, "star")


class TestStabilizerCircuitLookup_6_ladder(TestStabilizerCircuitLookupBase):
    def test_verify_stabilizer_6_ladder(self):
        self.verify_stabilizer_for_all(6, "ladder")

    def test_verify_lc_class_6_ladder(self):
        self.verify_lc_class_for_all(6, "ladder")

    def test_verify_state_6_ladder(self):
        self.verify_state_for_all(6, "ladder")

    def test_verify_connectivity_6_ladder(self):
        self.verify_connectivity_for_all(6, "ladder")


class TestStabilizerCircuitLookup_6_E(TestStabilizerCircuitLookupBase):
    def test_verify_stabilizer_6_E(self):
        self.verify_stabilizer_for_all(6, "E")

    def test_verify_lc_class_6_E(self):
        self.verify_lc_class_for_all(6, "E")

    def test_verify_state_6_E(self):
        self.verify_state_for_all(6, "E")

    def test_verify_connectivity_6_E(self):
        self.verify_connectivity_for_all(6, "E")


class TestStabilizerCircuitLookup_6_H(TestStabilizerCircuitLookupBase):
    def test_verify_stabilizer_6_H(self):
        self.verify_stabilizer_for_all(6, "H")

    def test_verify_lc_class_6_H(self):
        self.verify_lc_class_for_all(6, "H")

    def test_verify_state_6_H(self):
        self.verify_state_for_all(6, "H")

    def test_verify_connectivity_6_H(self):
        self.verify_connectivity_for_all(6, "H")


class TestStabilizerCircuitLookup_6_Q(TestStabilizerCircuitLookupBase):
    def test_verify_stabilizer_6_Q(self):
        self.verify_stabilizer_for_all(6, "Q")

    def test_verify_lc_class_6_Q(self):
        self.verify_lc_class_for_all(6, "Q")

    def test_verify_state_6_Q(self):
        self.verify_state_for_all(6, "Q")

    def test_verify_connectivity_6_Q(self):
        self.verify_connectivity_for_all(6, "Q")


class TestVerifyCostAndDepthInfo(TestStabilizerCircuitLookupBase):

    def test_verify_cost_and_depth(self):
        for num_qubits, connectivity in get_available_connectivities():
            self.verify_cost_and_depth_for_all(num_qubits, connectivity)


class TestStabilizerCircuitInfo(unittest.TestCase):

    def test_circuit_info(self):
        info = StabilizerCircuitInfo(4, "23:4:3:cx0,1 cx2,3 cx3,2 cx2,1 h0")
        self.assertEqual(info.num_qubits, 4)
        self.assertEqual(info.graph_id, 23)
        self.assertEqual(info.cost, 4)
        self.assertEqual(info.depth, 3)
        self.assertEqual(info.circuit_string, "cx0,1 cx2,3 cx3,2 cx2,1 h0")
        circuit = info.parse_circuit()


class TestMubLookupBase(unittest.TestCase):

    def verify_connectivity(self, num_qubits, connectivity):
        mub_info = mub_circuit_lookup(num_qubits, connectivity)

        connectivity_graph = get_connectivity_graph(num_qubits, connectivity)
        for circuit in mub_info.circuits:
            self.assertTrue(respects_connectivity(circuit, connectivity_graph))

    def verify_pauli_group_completeness(self, mub_info):
        num_qubits = mub_info.num_qubits
        paulis = set()
        for mub in mub_info.mubs:
            stabilizer = Stabilizer(mub)
            R, S = stabilizer.expand()
            for i in range(1, 2**num_qubits):
                col = np.concatenate([R[:, i], S[:, i]])
                paulis.add(str(col))
        self.assertEqual(len(paulis), 4**num_qubits - 1)

    def verify_circuit_correctness(self, mub_info):
        """Check that each circuit actually diagonalizes the stabilizer it promises to"""
        i = 0
        for mub, circuit in zip(mub_info.mubs, mub_info.circuits):
            if (not Stabilizer(mub).is_equivalent_mod_phase(Stabilizer(circuit.inverse()))):
                print(Stabilizer(mub), Stabilizer(circuit.inverse()), i)
            self.assertTrue(Stabilizer(mub).is_equivalent_mod_phase(Stabilizer(circuit.inverse())))
            i += 1


class TestAllMubs(TestMubLookupBase):

    def test_connectivity(self):
        for num_qubits, connectivity in get_available_connectivities():
            with self.subTest(num_qubits=num_qubits, connectivity=connectivity):
                self.verify_connectivity(num_qubits, connectivity)

    def test_pauli_group_completeness(self):
        for num_qubits, connectivity in get_available_connectivities():
            with self.subTest(num_qubits=num_qubits, connectivity=connectivity):
                self.verify_pauli_group_completeness(mub_circuit_lookup(num_qubits, connectivity))

    def test_circuit_correctness(self):
        for num_qubits, connectivity in get_available_connectivities():
            with self.subTest(num_qubits=num_qubits, connectivity=connectivity):
                self.verify_circuit_correctness(mub_circuit_lookup(num_qubits, connectivity))
