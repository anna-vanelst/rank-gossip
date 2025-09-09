import numpy as np
from src.graph import generate_graph
from src.utils import compute_connectivity
from src.utils import compute_p_value
from src.utils import Transform
from scipy import stats
from scipy.stats import rankdata
from omegaconf import OmegaConf


def load_data_arange(config):
    """Generate data as an array from 1 to n, with optional outliers. Also generate a graph.
    Returns the data, the graph, and None (no flags)."""
    n = config.n
    data = np.arange(1, n + 1)
    eps = OmegaConf.select(config, "eps", default=0.0)
    n_outliers = int(eps * n)
    if n_outliers > 0:
        indices = np.random.choice(n, n_outliers, replace=False)
        data[indices] = config.outlier * data[indices]
    graph = generate_graph(n=n, type=config.graph, seed=config.seed)
    print(f"Connectivity: {compute_connectivity(graph):.2e}")
    return data, graph, None


def load_data_groups(config):
    """Generate heavy-tailed data from two groups, with optional transformation. Also generate a graph.
    Returns the data, the graph, and the flags indicating group membership."""
    np.random.seed(config.seed if hasattr(config, "seed") else 0)
    n = config.n
    n1 = n // 2
    n2 = n - n1
    # Generate heavy-tailed group data
    group1 = stats.cauchy.rvs(loc=0.8, scale=1.0, size=n1)  # shifted group
    group2 = stats.cauchy.rvs(loc=0.0, scale=1.0, size=n2)
    data = np.concatenate([group1, group2])
    # Compute ranks
    ranks = rankdata(data)
    ranks1 = ranks[:n1]
    # Apply optional transformation
    transform = Transform(n, type="identity")
    ranks = transform.apply(ranks)
    ranks1 = transform.apply(ranks1)
    # Rank sum test statistic
    statistic = np.sum(ranks1)
    flags = np.concatenate([np.ones(n1), np.zeros(n2)])
    estimated_stat = n * np.mean(ranks * flags)
    # Compute p-value
    true_p_value = compute_p_value(statistic, n1, n2)
    print("mean", np.mean(ranks1))
    print("Rank sum statistic:", statistic)
    print("Estimated statistic", estimated_stat)
    print("P-value:", true_p_value)
    # Generate graph and edges
    graph = generate_graph(n=n, type=config.graph, seed=config.seed)
    print(f"Connectivity: {compute_connectivity(graph):.2e}")

    return data, graph, flags
