import numpy as np
from typing import List, Tuple, Union


class Graph:
    """
    A basic class for representing undirected, simple (no double edges, no edge from a vertex to itself) 
    graphs through an adjacency matrix. 
    """

    def __init__(self, data: Union[int, np.ndarray]):
        """Initialize a graph without edges with n vertices or with a 
        symmetric adjacency matrix. The entries of the letter should 
        consist of only 0 and 1 and the main diagonal should be all 0.  

        Note that the adjacency is internally stored using the data type
        np.int8, so to avoid a conversion you can pass in a numpy array
        of that type. 


        Parameters
        ----------
        data : int | np.ndarray
            Number of vertices | adjacency matrix
        """
        if isinstance(data, int):
            self.num_vertices = data
            self.adjacency_matrix = np.zeros((self.num_vertices, self.num_vertices), dtype=np.int8)
        elif isinstance(data, np.ndarray):
            assert data.shape[0] == data.shape[1], "Input adjacency matrix is not square"
            if data.dtype == np.int8:
                self.adjacency_matrix = data
            else:
                self.adjacency_matrix = data.astype(np.int8)
            self.adjacency_matrix &= 1
            assert np.array_equal(self.adjacency_matrix.T, self.adjacency_matrix), "Input adjacency matrix is not symmetric"
            self.num_vertices = self.adjacency_matrix.shape[0]
        else:
            assert False, "Unsupported input data format. Please provide either the number of vertices or a 2-dimensional symmetrical adjacency matrix"

    def copy(self):
        result = Graph(self.num_vertices)
        result.adjacency_matrix = self.adjacency_matrix.copy()
        return result

    def has_edge(self, vertex1: int, vertex2: int) -> bool:
        """Returns true if the graph has an edge between the given vertices. """
        return self.adjacency_matrix[vertex1, vertex2] == 1

    def add_edge(self, vertex1: int, vertex2: int):
        """Add an edge between the given vertices. """
        # assert 0<vertex1<self.num_vertices, f"Graph.add_edge"
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
        """Remove edge between vertex1 and vertex2. """
        self.adjacency_matrix[vertex1, vertex2] = 0
        self.adjacency_matrix[vertex2, vertex1] = 0

    def remove_all_edges_to(self, vertex):
        """Remove all edges where give vertex is one end of the edge. """
        for i in range(self.num_vertices):
            self.remove_edge(i, vertex)

    def clear(self):
        """Remove all edges from the graph. """
        self.adjacency_matrix.fill(0)

    def edge_count(self) -> int:
        """Count the number of edges in the graph. """
        return self.adjacency_matrix.sum() // 2

    def get_edges(self) -> List[Tuple[int, int]]:
        """Get a list of tuples containing the vertices for each edge in the graph. """
        edges = []
        for i in range(self.num_vertices-1):
            for j in range(i+1, self.num_vertices):
                if self.has_edge(i, j):
                    edges.append((i, j))
        return edges

    def draw(self, filename=None, show=False, size=100, point_size=10, show_vertex_labels: bool = True, first_vertex_label: int = 1):
        """
        Draw a graph using matplotlib. If filename is provided, the result
        graphic is storedat the path given by the filename. 

        When drawn, the zeroth vertex vertex is drawn at the topmost position
        and from there the order is clockwise. 

        Parameters
        ----------
        filename : str, optional
            If specified, the resulting graphics is saved as an image to this location, by default None
        show : bool, optional
            If set to true, plt.show() is called, usually showing the graphic in a new window, by default False
        size : int, optional
            Graphic size in px , by default 100
        point_size : int, optional
            Vertex size in px, by default 10
        show_vertex_labels : bool, optional
            If true, vertex index labels are shown, by default True
        first_vertex_label : int, optional
            The vertex index to start with, e.g. 0 or 1, by default 1

        Returns
        -------
        figure
            A matplotlib figure
        """
        from .graph_draw import draw_graph
        return draw_graph(self, filename=filename, show=show, size=size, point_size=point_size, show_vertex_labels=show_vertex_labels, first_vertex_label=first_vertex_label)

    @staticmethod
    def fully_connected(num_vertices: int):
        """Create a fully connected graph with n vertices. """
        graph = Graph(num_vertices)
        graph.adjacency_matrix.fill(1)
        for i in range(num_vertices):
            graph.adjacency_matrix[i, i] = 0
        return graph

    @staticmethod
    def star(num_vertices: int, center: int = 0):
        """Create a star graph with n vertices and center being the star center vertex. """
        graph = Graph(num_vertices)
        graph.adjacency_matrix[center, :] ^= 1
        graph.adjacency_matrix[:, center] ^= 1
        return graph

    @staticmethod
    def linear(num_vertices: int):
        """Create a linear graph (connected path graph) from vertex 0 to (n-1). """
        graph = Graph(num_vertices)
        for i in range(num_vertices - 1):
            graph.add_edge(i, i+1)
        return graph

    @staticmethod
    def cycle(num_vertices: int):
        """Create a cycle graph (connected path graph) from vertex 0 to (n-1) and back to 0. """
        graph = Graph.linear(num_vertices)
        graph.add_edge(0, num_vertices - 1)
        return graph

    @staticmethod
    def pusteblume(num_vertices: int):
        """Create a Pusteblume graph star from (n-2) vertices and the remaining 2 vertices both connected to one vertex from the star subgraph.  """
        assert num_vertices >= 5, "The pusteblume graph is only possible for at least 5 vertices"
        graph = Graph(num_vertices)
        for i in range(1, 4):
            graph.add_edge(0, i)
        for i in range(4, num_vertices):
            graph.add_edge(3, i)
        return graph

    def local_complementation(self, vertex: int):
        """
        Apply a local complementation inplace at given vertex. A local complementation
        takes the neighborhood N(i) of a vertex i and replaces the subgraph which is given by the 
        edges between vertices in N(i) by its graph complement. 
        """
        col = self.adjacency_matrix[:, vertex].reshape(1, self.num_vertices)
        self.adjacency_matrix ^= col.T @ col
        for i in range(self.num_vertices):
            self.adjacency_matrix[i, i] = 0

    def local_complemented(self, vertex: int):
        """Return a new graph which results from performing local complementation at given vertex. """
        result = self.copy()
        result.local_complementation(vertex)
        return result

    def swap(self, vertex1: int, vertex2: int):
        """Swap two vertices in the graph (transform edges according to the new vertex positions). """
        self.adjacency_matrix[[vertex1, vertex2]] = self.adjacency_matrix[[vertex2, vertex1]]
        self.adjacency_matrix[:, [vertex1, vertex2]] = self.adjacency_matrix[:, [vertex2, vertex1]]

    def __eq__(self, value) -> bool:
        return self.num_vertices == value.num_vertices and np.array_equal(self.adjacency_matrix, value.adjacency_matrix)

    def compress(self) -> int:
        """
        Compress the graphs adjacency matrix into an integer where each
        bit in the integer represents one edge of the graph. 

        I.e. it contains the graph in compressed form as a bitstring integer
        encoding one half of the adjacency matrix beginning with the least 
        significant bit. For example take a 5×5 adjacency matrix, then the bit 
        positions for each entry in the upper half are
            - 0 1 2 3
              - 4 5 6
                - 7 8
                  - 9
                    -

        This way a graph with the adjacency matrix
           0 1 1 0 1
           1 0 1 0 0
           1 1 0 1 1  
           0 0 1 0 0
           1 0 1 0 0
        becomes 0b0110011011
        Returns
        -------
        int
            graph id
        """
        n = self.num_vertices
        code = 0
        index = 0
        for i in range(self.num_vertices - 1):
            for j in range(i + 1, self.num_vertices):
                if self.has_edge(i, j):
                    code |= (1 << index)
                index += 1
        return code

    @staticmethod
    def decompress(num_vertices: int, id: int) -> "Graph":
        graph = Graph(num_vertices)
        index = 0
        for i in range(num_vertices - 1):
            for j in range(i + 1, num_vertices):
                if id & (1 << index):
                    graph.add_edge(i, j)
                index += 1
        return graph
