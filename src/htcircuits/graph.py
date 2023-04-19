import numpy as np
from typing import List, Tuple


class Graph:
    """A basic class for representing undirected, simple (no double edges, no edge from a vertex to itself) 
    graphs through an adjacency matrix. 

    """

    def __init__(self, num_vertices: int):
        """Initialize a graph without edges with n vertices
        """
        self.num_vertices = num_vertices
        self.adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=np.int8)

    def copy(self):
        result = Graph(self.num_vertices)
        result.adjacency_matrix = self.adjacency_matrix.copy()
        return result

    def has_edge(self, vertex1: int, vertex2: int) -> bool:
        """Returns true if the graph has an edge between the given vertices. 
        """
        return self.adjacency_matrix[vertex1, vertex2] == 1

    def add_edge(self, vertex1: int, vertex2: int):
        """Add an edge between the given vertices
        """
        if vertex1 == vertex2:
            return
        self.adjacency_matrix[vertex1, vertex2] = 1
        self.adjacency_matrix[vertex2, vertex1] = 1

    def add_path(self, path: List[int]):
        """Add a path of edges through the given vertices. 

        Parameters
        ----------
        path : List[int]
            Vertex list describing the path
        """
        if len(path) < 2:
            return
        previous_vertex = path[0]
        for vertex in path[1:]:
            self.add_edge(previous_vertex, vertex)
            previous_vertex = vertex

    def remove_edge(self, vertex1: int, vertex2: int):
        """Remove edge between vertex1 and vertex2
        """
        self.adjacency_matrix[vertex1, vertex2] = 0
        self.adjacency_matrix[vertex2, vertex1] = 0

    def remove_all_edges_to(self, vertex):
        """Remove all edges where give vertex is one end of the edge
        """
        for i in range(self.num_vertices):
            self.remove_edge(i, vertex)

    def clear(self):
        """Remove all edges from the graph
        """
        self.adjacency_matrix.fill(0)

    def edge_count(self) -> int:
        """Count the number of edges in the graph
        """
        return self.adjacency_matrix.sum() // 2

    def get_edges(self) -> List[Tuple[int]]:
        """Get a list of tuples containing the vertices for each edge in the graph
        """
        edges = []
        for i in range(self.num_vertices-1):
            for j in range(i+1, self.num_vertices):
                if self.has_edge(i, j):
                    edges.append((i, j))
        return edges

    def draw(self, filename=None, show=False, size=100, point_size=10):
        """Draw a graph using matplotlib. If filename is provided, the result graphic is stored
        at the path given by the filename. 

        Parameters
        ----------
        filename : str, optional
            If given, the graphic is stored here, by default None
        show : bool, optional
            Whether to show the graph in a separate wi, by default False
        size : int, optional
            Graphic size in px , by default 100
        point_size : int, optional
            Vertex size in px, by default 10

        Returns
        -------
        figure
            A matplotlib figure
        """
        from .graph_draw import draw_graph
        return draw_graph(self, filename=filename, show=show, size=size, point_size=point_size)

    @staticmethod
    def fully_connected(num_vertices: int):
        """Create a fully connected graph with n vertices.
        """
        graph = Graph(num_vertices)
        graph.adjacency_matrix.fill(1)
        for i in range(num_vertices):
            graph.adjacency_matrix[i, i] = 0
        return graph

    @staticmethod
    def star(num_vertices: int, center: int = 0):
        """Create a star graph with n vertices and center being the star center vertex.
        """
        graph = Graph(num_vertices)
        graph.adjacency_matrix[center, :] ^= 1
        graph.adjacency_matrix[:, center] ^= 1
        return graph

    @staticmethod
    def linear(num_vertices: int):
        """Create a linear graph (connected path graph) from vertex 0 to (n-1).
        """
        graph = Graph(num_vertices)
        for i in range(num_vertices - 1):
            graph.add_edge(i, i+1)
        return graph

    @staticmethod
    def cycle(num_vertices: int):
        """Create a cycle graph (connected path graph) from vertex 0 to (n-1) and back to 0.
        """
        graph = Graph.linear(num_vertices)
        graph.add_edge(0, num_vertices - 1)
        return graph

    @staticmethod
    def pusteblume(num_vertices: int):
        """Create a Pusteblume graph star from (n-2) vertices and the remaining 2 vertices both connected to one vertex from the star subgraph. 
        """
        assert num_vertices >= 5, "The pusteblume graph is only possible for at least 5 vertices"
        graph = Graph(num_vertices)
        for i in range(1, 4):
            graph.add_edge(0, i)
        for i in range(4, num_vertices):
            graph.add_edge(3, i)
        return graph

    def local_complementation(self, vertex: int):
        """Apply a local complementation inplace at given vertex. A local complementation
        takes the neighborhood N(i) of a vertex i and replaces the subgraph which is given by the 
        edges between vertices in N(i) by its graph complement. 
        """
        col = self.adjacency_matrix[:, vertex].reshape(1, self.num_vertices)
        self.adjacency_matrix ^= col.T @ col
        for i in range(self.num_vertices):
            self.adjacency_matrix[i, i] = 0

    def local_complemented(self, vertex: int):
        """Return a new graph which results from performing local complementation at given vertex
        """
        result = self.copy()
        result.local_complementation(vertex)
        return result

    def swap(self, vertex1: int, vertex2: int):
        """Swap two vertices in the graph (transform edges according to the new vertex positions).
        """
        self.adjacency_matrix[[vertex1, vertex2]] = self.adjacency_matrix[[vertex2, vertex1]]
        self.adjacency_matrix[:, [vertex1, vertex2]] = self.adjacency_matrix[:, [vertex2, vertex1]]

    def __eq__(self, value) -> bool:
        return self.num_vertices == value.num_vertices and np.array_equal(self.adjacency_matrix, value.adjacency_matrix)

        # Graph.star(5, 0)
