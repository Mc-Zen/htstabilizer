from src.htcircuits.find_local_clifford_layer import *
from src.htcircuits.stabilizer import Stabilizer

from tests.random_stabilizer import random_stabilizer

import unittest


class TestFindLocalCliffordLayer(unittest.TestCase):

    def test_as(self):
        s = Stabilizer(["ZY", "IY"])
        A = find_local_clifford_layer(s.R, s.S, Graph(2))
        assert A is not None
        self.assertTrue(check_LC(s.R, s.S, Graph(2), A))
        local_clifford_layer_to_circuit(A)

        s = Stabilizer(["XZZ", "ZYZ", "ZZX"])
        A = find_local_clifford_layer(s.R, s.S, Graph.linear(3))
        assert A is not None
        self.assertTrue(check_LC(s.R, s.S, Graph.linear(3), A))
        local_clifford_layer_to_circuit(A)

    def test_random(self):
        for i in range(1000):
            graph = Graph.linear(3)
            stabilizer = random_stabilizer(graph)
            A = find_local_clifford_layer(stabilizer.R, stabilizer.S, graph)
            assert A is not None
            self.assertTrue(check_LC(stabilizer.R, stabilizer.S, graph, A))
            local_clifford_layer_to_circuit(A)

        for i in range(1000):
            graph = Graph.pusteblume(5)
            stabilizer = random_stabilizer(graph)
            A = find_local_clifford_layer(stabilizer.R, stabilizer.S, graph)
            assert A is not None
            self.assertTrue(check_LC(stabilizer.R, stabilizer.S, graph, A))
            local_clifford_layer_to_circuit(A)

        for i in range(100):
            graph = Graph.star(7)
            stabilizer = random_stabilizer(graph)
            A = find_local_clifford_layer(stabilizer.R, stabilizer.S, graph)
            assert A is not None
            self.assertTrue(check_LC(stabilizer.R, stabilizer.S, graph, A))
            local_clifford_layer_to_circuit(A)

    def test_empty_graph_case(self):
        graph = Graph(3)
        stabilizer = Stabilizer(["ZII", "IZI", "IIZ"])
        A = find_local_clifford_layer(stabilizer.R, stabilizer.S, graph)
        assert A is not None
        self.assertTrue(check_LC(stabilizer.R, stabilizer.S, graph, A))
        local_clifford_layer_to_circuit(A)
