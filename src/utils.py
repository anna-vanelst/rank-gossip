import numpy as np
import networkx as nx
from scipy.stats import norm


def wn(n, r, alpha=0.1, normalize=True):
    """Compute the weights for the trimmed mean."""
    weights = np.zeros_like(r, dtype=float)
    m = int(alpha * n)
    mask = (m + 0.5 <= r) & (r < n - m + 0.5)
    if n - 2 * m > 0:
        if normalize:
            weights[mask] = n / (n - 2 * m)
        else:
            weights[mask] = 1
    return weights


class Transform:
    """Class for applying transformations to ranks for rank statistics."""

    def __init__(self, n, type="identity"):
        self.n = n
        self.type = type

    def apply(self, ranks):
        """Apply the transformation to the ranks."""
        if self.type == "van_der_waerden":
            normalized_ranks = ranks / (self.n + 1)
            scores = norm.ppf(normalized_ranks)
            return scores
        elif self.type == "identity":
            return ranks
        else:
            raise ValueError(f"Unknown transformation type: {self.type}")


def compute_connectivity(graph):
    """Compute lambda_2/|E| for a given graph."""
    m = graph.number_of_edges()
    laplacian = nx.laplacian_matrix(graph).toarray()
    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
    lambda_2 = eigenvalues[1]
    return lambda_2 / m


def compute_weighted_connectivity(edges, weights):
    """Compute lambda_2 for a weighted graph given its edges and weights."""
    G = nx.Graph()
    for (u, v), w in zip(edges, weights):
        G.add_edge(u, v, weight=w)
    laplacian = nx.laplacian_matrix(G, weight="weight").toarray()
    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
    lambda_2 = eigenvalues[1]
    return lambda_2


def compute_p_value(statistic, n1, n2):
    """Compute the p-value for the rank sum test statistic."""
    mean = n1 * (n1 + n2 + 1) / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_score = (statistic - mean) / sigma
    # For a two-tailed test
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return p_value
