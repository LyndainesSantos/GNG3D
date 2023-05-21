import json
import networkx as nx
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_gng3d(path_es_vs):
    with open(path_es_vs+"/es.json", 'r') as openfile:
        graph_in = json.load(openfile)
    with open(path_es_vs+"/vs.json", 'r') as openfile:
        graph_in_vs = json.load(openfile)

    # G = nx.Graph()
    # G.add_edges_from(graph["edges"].itervalues())
    graph = np.array(eval(graph_in))
    # graph_vs = np.array(eval(graph_in_vs))
    fig = plt.figure(figsize = (50,50))
    ax = plt.axes(projection='3d')
    ax.grid()

    for i in range(0, len(graph) - 1, 2):
        x_values = [graph[i, 0], graph[i + 1, 0]]
        y_values = [graph[i, 1], graph[i + 1, 1]]
        z_values = [graph[i, 2], graph[i + 1, 2]]
        plt.plot(x_values, y_values, z_values, color='b')

    ax.scatter(graph[:,0], graph[:,1], graph[:,2], c = 'g', s = 5)
    # ax.scatter(graph_vs[:,0], graph_vs[:,1], graph_vs[:,2], c = 'g', s = 5)
    ax.set_title('3D Scatter Plot')

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    plt.show()