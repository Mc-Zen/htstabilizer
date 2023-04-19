from src.htcircuits.find_local_clifford_layer import *
from src.htcircuits.stabilizer import Stabilizer


import unittest


def random_stabilizer(graph: Graph):
    stabilizer = Stabilizer(graph)
    A = generate_local_clifford_symplectic_from_id(np.random.randint(0, 6, graph.num_vertices))
    R_prime = f2.add(f2.mat_mul(A[0], stabilizer.R), f2.mat_mul(A[1], stabilizer.S))
    S_prime = f2.add(f2.mat_mul(A[2], stabilizer.R), f2.mat_mul(A[3], stabilizer.S))
    stabilizer.R = R_prime
    stabilizer.S = S_prime
    return stabilizer


class TestFindLocalCliffordLayer(unittest.TestCase):

    def test_as(self):
        s = Stabilizer(["ZY", "IY"])
        A = find_local_clifford_layer(s.R, s.S, Graph(2))
        self.assertTrue(check_LC(s.R, s.S, Graph(2), A))

        s = Stabilizer(["XZZ", "ZYZ", "ZZX"])
        A = find_local_clifford_layer(s.R, s.S, Graph.linear(3))
        self.assertTrue(check_LC(s.R, s.S, Graph.linear(3), A))

    def test_random(self):
        for i in range(1000):
            graph = Graph.linear(3)
            stabilizer = random_stabilizer(graph)
            A = find_local_clifford_layer(stabilizer.R, stabilizer.S, graph)
            self.assertTrue(check_LC(stabilizer.R, stabilizer.S, graph, A))

        for i in range(1000):
            graph = Graph.pusteblume(5)
            stabilizer = random_stabilizer(graph)
            A = find_local_clifford_layer(stabilizer.R, stabilizer.S, graph)
            self.assertTrue(check_LC(stabilizer.R, stabilizer.S, graph, A))

        for i in range(100):
            graph = Graph.star(7)
            stabilizer = random_stabilizer(graph)
            A = find_local_clifford_layer(stabilizer.R, stabilizer.S, graph)
            self.assertTrue(check_LC(stabilizer.R, stabilizer.S, graph, A))
