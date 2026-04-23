import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_shortest_path_distribution(degrees, title="Shortest Path Length Distribution", relative=True):
    degrees = np.array(degrees, dtype=int)

    if np.sum(degrees) % 2 != 0:
        degrees[0] += 1

    graph = nx.configuration_model(degrees)

    G = nx.Graph(graph)
    G.remove_edges_from(nx.selfloop_edges(G))

    largest_cc = max(nx.connected_components(G), key=len)
    G_core = G.subgraph(largest_cc)

    path_lengths = dict(nx.shortest_path_length(G_core))

    lengths = []
    for source, targets in path_lengths.items():
        for target, length in targets.items():
            if source < target:
                lengths.append(length)

    if not lengths:
        return

    unique_lengths, counts = np.unique(lengths, return_counts=True)

    if relative:
        y_values = counts / counts.sum()
        ylabel = 'Relative Frequency'
    else:
        y_values = counts
        ylabel = 'Frequency (Count)'

    plt.figure(figsize=(10, 6))
    plt.bar(unique_lengths, y_values, color='skyblue', edgecolor='black', zorder=2)
    plt.xlabel('Shortest Path Length')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(unique_lengths)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    plt.show()
