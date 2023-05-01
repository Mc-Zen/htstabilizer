from .graph import Graph
import numpy as np


def draw_graph(graph: Graph, filename=None, show=False, size=100, point_size=10, show_vertex_labels: bool = True, first_vertex_label: int = 1):
    """Draw a graph object using matplotlib

    Parameters
    ----------
    graph : Graph
        graph object to draw
    filename : str, optional
        If specified, the resulting graphics is saved as an image to this location, by default None
    show : bool, optional
        If set to true, plt.show() is called, usually showing the graphic in a new window, by default False
    size : int, optional
        Figure size in px, by default 100
    point_size : int, optional
        Vertex size in px, by default 10
    show_vertex_labels : bool, optional
        If true, vertex index labels are shown, by default True
    first_vertex_label : int, optional
        The vertex index to start with, e.g. 0 or 1, by default 1

    Returns
    -------
    figure
        Matplotlib figure
    """
    from matplotlib import pyplot as plt
    figure = plt.figure()
    px = 1 / plt.rcParams["figure.dpi"]
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.set_aspect(1)
    ax.margins(point_size * px * (2 if show_vertex_labels else 1))

    xcoords, ycoords = [], []
    num_vertices = graph.num_vertices
    for i in range(num_vertices):
        angle = 2 * np.pi * i / num_vertices
        xcoords.append(np.sin(angle))
        ycoords.append(np.cos(angle))
        if show_vertex_labels:
            ax.text(1.4*xcoords[-1], 1.4*ycoords[-1], str(i+first_vertex_label), horizontalalignment='center', verticalalignment='center')

    edges = graph.get_edges()

    for vertex1, vertex2 in edges:  # type: ignore
        ax.plot([xcoords[vertex1], xcoords[vertex2]], [ycoords[vertex1], ycoords[vertex2]], "k-")

    # add artifical padding for the 2-vertex case
    if num_vertices == 2:
        ax.plot([-1, 1], [0, 0], alpha=0)

    ax.plot(xcoords, ycoords, "k.", markersize=point_size)
    figure.set_size_inches(size * px, size * px)
    # figure.tight_layout(rect=[0, 0,1, 1])

    if show:
        plt.show()

    if filename:
        figure.savefig(filename, bbox_inches="tight")

    if plt.rcParams['backend'] in ["module://ipykernel.pylab.backend_inline", "module://matplotlib_inline.backend_inline", "nbAgg"]:
        plt.close(figure)
    return figure
