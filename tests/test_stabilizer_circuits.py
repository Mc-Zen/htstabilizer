from src.htstabilizer.lc_classes import *
from src.htstabilizer.stabilizer_circuits import *

from tests.random_stabilizer import random_stabilizer

import unittest


class TestHTStabilizer(unittest.TestCase):

    def test_get_connectivity_graph(self):
        get_connectivity_graph(3, "all")
        get_connectivity_graph(3, "linear")
        get_connectivity_graph(4, "all")
        get_connectivity_graph(4, "linear")
        get_connectivity_graph(5, "all")
        get_connectivity_graph(5, "linear")
        get_connectivity_graph(5, "T")
        get_connectivity_graph(5, "star")

        # get_connectivity_graph(5, "T").draw(show=True)

    def test_special_cases(self):
        stabilizer = Stabilizer(["ZIIIX", "IZIII", "IIXXI", "IIZYI", "YIIIY"])
        qc = get_preparation_circuit(stabilizer, "all")

    def verify_random_stabilizers(self, num_qubits, connectivity, num):
        LCClasses = [LCClass2, LCClass3, LCClass4, LCClass5]
        cls = LCClasses[num_qubits - 2]
        for id in range(cls.count()):
            for i in range(num):
                stabilizer = random_stabilizer(cls(id).get_graph())
                # stabilizer = Stabilizer(cls(i).get_graph())
                qc = get_preparation_circuit(stabilizer, connectivity)
                self.assertTrue(stabilizer.is_equivalent(Stabilizer(qc)))

    def test_basic(self):
        stabilizer = Stabilizer(["ZII", "IZI", "IIZ"], True)
        qc = get_preparation_circuit(stabilizer, "all")
        self.assertTrue(stabilizer.is_equivalent(Stabilizer(qc)))

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
