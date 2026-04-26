import numpy as np
import networkx as nx
import scipy.sparse as sp


def generate_network(degree_sequence, sigma, gamma, mu):
    sequence = [int(round(s)) for s in degree_sequence]
    if sum(sequence) % 2 != 0:
        sequence[0] += 1

    G_multi = nx.configuration_model(sequence)
    G = nx.Graph(G_multi)
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_numpy_array(G)

    M = np.random.normal(0, sigma, (len(sequence), len(sequence)))
    S = (M + M.T) / np.sqrt(2)
    V = (M - M.T) / np.sqrt(2)
    Alpha = np.sqrt(1 + gamma) * S + np.sqrt(1 - gamma) * V + mu

    W = A * Alpha
    return W, G


def generate_matrix(
    degree_sequence: list[int],
    C: float,
    mu: float,
    sigma: float,
):
    """Build a sparse interaction matrix using the configuration model.

    Edge weights: alpha_ij = mu/C + (sigma/sqrt(C)) * z_ij, z_ij ~ N(0,1).
    Self-loops and multi-edges are removed. Diagonal is zero.

    Args:
        degree_sequence: Integer degree of each node. Sum must be even.
        C: Mean degree used in weight formula.
        mu: Mean interaction strength parameter.
        sigma: Std of interaction strength fluctuations.

    Returns:
        scipy.sparse.csr_array of shape (N, N).

    Raises:
        ValueError: If the sum of degree_sequence is odd.
    """
    if sum(degree_sequence) % 2 != 0:
        raise ValueError("Sum of degree_sequence must be even.")

    mu_effective = mu

    G_multi = nx.configuration_model(degree_sequence)
    G = nx.Graph(G_multi)
    G.remove_edges_from(nx.selfloop_edges(G))

    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)

    rows, cols = A.nonzero()
    z = np.random.normal(0.0, 1.0, len(rows))
    A.data = (mu_effective / C) + (sigma / np.sqrt(C)) * z
    A.setdiag(0.0) 
    A.eliminate_zeros()

    return A
