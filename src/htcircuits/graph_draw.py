from .graph import Graph
import numpy as np


def draw_graph(graph: Graph, filename=None, show=False, size=100, point_size=10):
    from matplotlib import pyplot as plt
    figure = plt.figure()
    px = 1 / plt.rcParams["figure.dpi"]
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.set_aspect(1)
    ax.margins(point_size * px)

    xcoords, ycoords = [], []
    num_vertices = graph.num_vertices
    for i in range(num_vertices):
        angle = 2 * np.pi * i / num_vertices
        xcoords.append(np.sin(angle))
        ycoords.append(np.cos(angle))

    edges = graph.get_edges()

    for vertex1, vertex2 in edges: # type: ignore
        ax.plot([xcoords[vertex1], xcoords[vertex2]], [ycoords[vertex1], ycoords[vertex2]], "k-")

    # add artifical padding for the 2-vertex case
    if num_vertices == 2:
        ax.plot([-1, 1], [0, 0], alpha=0)

    ax.plot(xcoords, ycoords, "k.", markersize=point_size)
    figure.set_size_inches(size * px, size * px)

    if show:
        plt.show()

    if filename:
        figure.savefig(filename, bbox_inches="tight")

    if plt.rcParams['backend'] in ["module://ipykernel.pylab.backend_inline", "module://matplotlib_inline.backend_inline", "nbAgg"]:
        plt.close(figure)
    return figure
