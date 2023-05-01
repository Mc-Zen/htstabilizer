from src.htcircuits.graph import Graph


import unittest
import numpy as np


class TestGraph(unittest.TestCase):
    def test_constructor(self):
        self.assertEqual(Graph(5).num_vertices, 5)
        self.assertEqual(Graph(89).num_vertices, 89)

        graph = Graph(np.array([[0, 1], [1, 0]]))
        self.assertEqual(graph.num_vertices, 2)

        graph = Graph(np.array([[0., 1.], [1., 0.]]))
        self.assertEqual(graph.adjacency_matrix.dtype, np.int8)
        self.assertTrue(np.array_equal(graph.adjacency_matrix, np.array([[0, 1], [1, 0]])))

    def test_edge_count(self):
        g = Graph(6)
        self.assertEqual(g.edge_count(), 0)
        g.add_edge(4, 5)
        g.add_edge(4, 5)
        self.assertEqual(g.edge_count(), 1)
        g.add_edge(3, 4)
        self.assertEqual(g.edge_count(), 2)
        g.clear()
        self.assertEqual(g.edge_count(), 0)

    def test_add_remove_edge(self):
        g = Graph(5)
        self.assertFalse(g.has_edge(1, 0))
        g.add_edge(1, 0)
        self.assertTrue(g.has_edge(1, 0))
        self.assertTrue(g.has_edge(0, 1))
        g.remove_edge(0, 1)
        self.assertFalse(g.has_edge(1, 0))
        self.assertFalse(g.has_edge(0, 1))

    def test_add_path(self):
        g = Graph(5)
        g.add_path([2, 3, 4, 1, 0])
        self.assertTrue(g.has_edge(2, 3))
        self.assertTrue(g.has_edge(3, 4))
        self.assertTrue(g.has_edge(4, 1))
        self.assertTrue(g.has_edge(1, 0))
        self.assertEqual(g.edge_count(), 4)

    def test_draw(self):
        g = Graph(5)
        self.assertTrue(g.draw() != None)
        # g.add_path([0,1,2,3])
        g.add_path([0, 1, 2, 4])
        g = Graph.star(5, 1)
        g.draw(show=True)

    def test_star(self):
        self.assertTrue(np.array_equal(Graph.star(3, 1).adjacency_matrix, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)))
        self.assertTrue(np.array_equal(Graph.star(3, 2).adjacency_matrix, np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.int8)))

    def test_fully_connected(self):
        self.assertTrue(np.array_equal(Graph.fully_connected(3).adjacency_matrix, np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)))

    def test_line(self):
        self.assertTrue(np.array_equal(Graph.fully_connected(3).adjacency_matrix, np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)))

    def test_local_complementation(self):
        g = Graph.star(5, 1)
        g.local_complementation(1)
        # print(g.adjacency_matrix)
        # g.draw(show=True)
        self.assertEqual(g, Graph.fully_connected(5))

    def test_swap(self):
        g = Graph(5)
        g.add_path([1, 0, 2, 3, 4])
        g.swap(0, 1)
        self.assertEqual(g, Graph.linear(5))

    def test_compress_decompress(self):
        self.assertEqual(Graph.decompress(9, Graph.star(9).compress()), Graph.star(9))
        self.assertEqual(Graph.star(9).compress(), 0b11111111)

        self.assertEqual(Graph.decompress(9, Graph.star(9, 4).compress()), Graph.star(9, 4))
        # self.assertEqual(Graph.star(9).compress(), 0b11111111)

        self.assertEqual(Graph.decompress(9, Graph.pusteblume(9).compress()), Graph.pusteblume(9))
        self.assertEqual(Graph.decompress(9, Graph.fully_connected(9).compress()), Graph.fully_connected(9))
        self.assertEqual(Graph.decompress(9, Graph(9).compress()), Graph(9))
        self.assertEqual(Graph.decompress(9, Graph.linear(9).compress()), Graph.linear(9))
