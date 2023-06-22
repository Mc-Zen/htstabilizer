from src.htstabilizer.stabilizer_circuits import *
from src.htstabilizer.lc_classes import *
from src.htstabilizer.connectivity_support import *
from src.htstabilizer.rotate_stabilizer_into_state import assert_same_state

from tests.random_stabilizer import random_stabilizer

import unittest

from qiskit.quantum_info import random_clifford


class TestHTStabilizer(unittest.TestCase):

    def test_special_cases(self):
        stabilizer = Stabilizer(["ZIIIX", "IZIII", "IIXXI", "IIZYI", "YIIIY"])
        qc = get_preparation_circuit(stabilizer, "all")

    def verify_random_stabilizers(self, num_qubits, connectivity, num):
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5, LCClass6]
        cls = LCClasses[num_qubits - 2]
        for id in range(1, cls.count()):
            for i in range(num):
                stabilizer = random_stabilizer(cls(id).get_graph())
                # stabilizer = Stabilizer(cls(i).get_graph())
                qc = get_preparation_circuit(stabilizer, connectivity)
                self.assertTrue(stabilizer.is_equivalent_mod_phase(Stabilizer(qc)))
                # print(stabilizer, Stabilizer(qc))
                self.assertTrue(stabilizer.is_equivalent(Stabilizer(qc)))

    def test_basic(self):
        stabilizer = Stabilizer(["ZII", "IZI", "IIZ"], True)
        qc = get_preparation_circuit(stabilizer, "all")
        self.assertTrue(stabilizer.is_equivalent_mod_phase(Stabilizer(qc)))

    def test_random_stabilizers_2_all(self):
        self.verify_random_stabilizers(2, "all", num=500)

    def test_random_stabilizers_3_all(self):
        self.verify_random_stabilizers(3, "all", num=500)

    def test_random_stabilizers_3_linear(self):
        self.verify_random_stabilizers(3, "linear", num=500)

    def test_random_stabilizers_4_all(self):
        self.verify_random_stabilizers(4, "all", num=100)

    def test_random_stabilizers_4_linear(self):
        self.verify_random_stabilizers(4, "linear", num=100)

    def test_random_stabilizers_4_star(self):
        self.verify_random_stabilizers(4, "star", num=100)

    def test_random_stabilizers_4_cycle(self):
        self.verify_random_stabilizers(4, "cycle", num=100)

    def test_random_stabilizers_5_all(self):
        self.verify_random_stabilizers(5, "all", num=20)

    def test_random_stabilizers_5_linear(self):
        self.verify_random_stabilizers(5, "linear", num=20)

    def test_random_stabilizers_5_star(self):
        self.verify_random_stabilizers(5, "star", num=20)

    def test_random_stabilizers_5_cycle(self):
        self.verify_random_stabilizers(5, "cycle", num=20)

    def test_random_stabilizers_5_T(self):
        self.verify_random_stabilizers(5, "T", num=20)

    def test_random_stabilizers_5_Q(self):
        self.verify_random_stabilizers(5, "Q", num=20)

    # def test_random_stabilizers_6_all(self):
    #     self.verify_random_stabilizers(6, "all", num=20)


class TestCliffordOptimization(unittest.TestCase):

    def test_random_clifford_optimization(self):
        for num_qubits, connectivity in get_available_connectivities():
            with self.subTest(num_qubits=num_qubits, connectivity=connectivity):
                for i in range(100):
                    cliff = random_clifford(num_qubits)
                    qc = cliff.to_circuit()
                    assert qc is not None
                    qc2 = compress_preparation_circuit(qc, connectivity)  # type: ignore
                    assert_same_state(qc, qc2)
