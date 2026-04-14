import numpy as np
import networkx as nx


def get_degree_sequence(N: int, C: float, topology: str = "regular") -> list[int]:
    """Return a degree sequence of length N with mean degree C.

    Args:
        N: Number of nodes.
        C: Mean degree (exact for regular, expected value for exponential).
        topology: "regular" or "exponential".

    Returns:
        List of integer degrees whose sum is even.

    Raises:
        ValueError: If topology is not recognised.
    """
    if topology == "regular":
        degrees = np.full(N, int(C))
    elif topology == "exponential":
        degrees = np.round(np.random.exponential(scale=C, size=N)).astype(int)
    else:
        raise ValueError(f"Unknown topology '{topology}'. Use 'regular' or 'exponential'.")

    if degrees.sum() % 2 != 0:
        degrees[np.random.randint(0, N)] += 1

    return degrees.tolist()


def compute_mu_c(degree_sequence: list[int], C: float) -> float:
    """Return the critical interaction strength mu_c = 1 / <g^2>.

    g_i = k_i / C is the normalised degree of node i.

    Args:
        degree_sequence: Integer degree of each node.
        C: Mean degree used to normalise.

    Returns:
        Critical mu value.
    """
    g = np.array(degree_sequence, dtype=float) / C
    return float(1.0 / np.mean(g ** 2))


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

    G_multi = nx.configuration_model(degree_sequence)
    G = nx.Graph(G_multi)
    G.remove_edges_from(nx.selfloop_edges(G))

    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)

    rows, cols = A.nonzero()
    z = np.random.normal(0.0, 1.0, len(rows))
    A.data = (mu / C) + (sigma / np.sqrt(C)) * z
    A.setdiag(0.0)
    A.eliminate_zeros()

    return A
