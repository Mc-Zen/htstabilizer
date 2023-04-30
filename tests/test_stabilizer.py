from qiskit import QuantumCircuit
from src.htcircuits.stabilizer import Stabilizer
from src.htcircuits.graph import Graph

import unittest
import numpy as np


class TestStabilizer(unittest.TestCase):

    def test_matrix_constructor(self):
        R = np.array([[1, 0, 0],
                      [0, 1, 1],
                      [0, 1, 1]])
        s = Stabilizer((R, np.eye(3)))
        self.assertTrue(np.array_equal(s.R, R))
        self.assertTrue(np.array_equal(s.S, np.eye(3, dtype=np.int8)))
        self.assertTrue(s.validate(), 3)

        with self.assertRaises(AssertionError):
            Stabilizer(["XXY", "IZX", "XYZ"], validate=True)

        with self.assertRaises(AssertionError):
            Stabilizer(["XXZ", "YYZ", "XXZ"], validate=True)

        with self.assertRaises(AssertionError):
            Stabilizer(["XXZ", "YYI", "ZZX"], validate=True)

        with self.assertRaises(AssertionError):
            R = np.array([[1, 0, 0],
                          [0, 1, 1],
                          [0, 0, 1]])
            s = Stabilizer((R, np.eye(3)), validate=True)

    def test_paulistring_constructor(self):
        s = Stabilizer(["YII", "IYX", "IXY"])
        X = np.array([[1, 0, 0],
                      [0, 1, 1],
                      [0, 1, 1]])
        self.assertTrue(np.array_equal(s.R, X))
        self.assertTrue(np.array_equal(s.S, np.eye(3, dtype=np.int8)))

    def test_graph_state_constructor(self):
        s = Stabilizer(Graph.star(4))
        R = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.assertTrue(np.array_equal(s.R, R))
        S = np.array([[0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0]])
        self.assertTrue(np.array_equal(s.S, S))

    def test_quantum_circuit_constructor(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        s = Stabilizer(qc)
        self.assertEqual(s, Stabilizer(["XII", "IZI", "IIZ"]))
        qc = QuantumCircuit(3)
        qc.cx(1, 2)
        qc.h(0)
        s = Stabilizer(qc)
        self.assertEqual(s, Stabilizer(["XII", "IZI", "IZZ"]))

    def test_equality(self):

        s1 = Stabilizer(["XZZ", "ZXI", "ZIX"])

        s2 = Stabilizer(Graph.star(3))

        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        S = np.array([[0, 1, 1],
                      [1, 0, 0],
                      [1, 0, 0]])
        s3 = Stabilizer((R, S))

        self.assertEqual(s1, s2)
        self.assertEqual(s2, s3)
        self.assertEqual(s1, s3)

    def test_expand(self):
        R = np.array([[1, 0, 0],
                      [0, 1, 1],
                      [0, 1, 1]])
        s = Stabilizer((R, np.eye(3)))
        X, Z = s.expand()
        self.assertTrue(np.array_equal(X, np.array([[0, 1, 0, 1, 0, 1, 0, 1],
                                                    [0, 0, 1, 1, 1, 1, 0, 0],
                                                    [0, 0, 1, 1, 1, 1, 0, 0]])))

        self.assertTrue(np.array_equal(Z, np.array([[0, 1, 0, 1, 0, 1, 0, 1],
                                                    [0, 0, 1, 1, 0, 0, 1, 1],
                                                    [0, 0, 0, 0, 1, 1, 1, 1]])))

    def test_is_qubit_entangled(self):
        s = Stabilizer(["XYI", "ZXX", "YZI"], validate=True)
        self.assertEqual(s.is_qubit_entangled(0), True)
        self.assertEqual(s.is_qubit_entangled(1), True)
        self.assertEqual(s.is_qubit_entangled(2), False)

    def test_is_equivalent(self):
        s = Stabilizer(["XYI", "ZXX", "YZI"], validate=True)
        self.assertTrue(s.is_equivalent(Stabilizer(["YZI", "ZXX", "XYI"])))
        self.assertTrue(s.is_equivalent(Stabilizer(["ZXX", "YZI", "XYI"])))
        self.assertTrue(s.is_equivalent(Stabilizer(["XYX", "YZI", "XYI"])))
        self.assertTrue(s.is_equivalent(Stabilizer(["XYX", "YZI", "ZXI"])))
        self.assertFalse(s.is_equivalent(Stabilizer(["YZI", "ZXX", "XXI"])))
