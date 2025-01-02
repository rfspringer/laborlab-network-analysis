import networkx as nx
from itertools import permutations, product
from scipy.stats import spearmanr
import numpy as np
from scipy import stats


def is_valid_network(graph: nx.DiGraph):
    if not isinstance(graph, nx.DiGraph):
        return False, "Must input NetworkX graph"
    elif not nx.is_directed(graph):
        return False, "Graph must be directed"
    else:
        return True


def is_linearly_independent(cycle, basis):
    for basis_cycle in basis:
        if set(cycle).intersection(set(basis_cycle)):
            return False
    return True


def get_basis(graph):
    all_cycles = list(nx.simple_cycles(graph))
    cycle_basis = []
    for cycle in all_cycles:
        if is_linearly_independent(cycle, cycle_basis):
            cycle_basis.append(cycle)
    return cycle_basis


def get_list_of_set_permutations(list_of_sets):
    set_permutations = [list(permutations(s)) for s in list_of_sets]
    product_permutations = product(*set_permutations)

    # Combine permutations and preserve the order of the sets
    result = [
        [item for sublist in perm for item in sublist] for perm in product_permutations
    ]

    return result



def calculate_rank_correlation(sorted_nodes, ranks_by_id):
    """
    Calculate Spearman rank correlation between true sorted order of wealth and results from sorting algorithm.
    :param sorted_nodes: List of sets of nodes sorted by the algorithm. Nodes in the same set are tied.
    :param ranks_by_id: Dictionary of node ranks by their IDs.
    :return: Spearman rank correlation between true order and sorting algorithm results, list used to calculate result.
    """

    # Flatten the sorted nodes, preserving order
    flat_sorted_nodes = [node for node_set in sorted_nodes for node in node_set]

    # Get the ranks of the nodes based on their true ranks
    list_ranks = [ranks_by_id[node] for node in flat_sorted_nodes]

    # Create a rank list where tied nodes have the average of their ranks
    true_ranks = []
    current_rank = 1
    for node_set in sorted_nodes:
        set_size = len(node_set)
        avg_rank = sum(range(current_rank, current_rank + set_size)) / set_size
        true_ranks.extend([avg_rank] * set_size)
        current_rank += set_size

    # Calculate Spearman rank correlation
    rho, _ = spearmanr(list_ranks, true_ranks)

    return rho, flat_sorted_nodes


def multivariate_pareto_dist(n_samples, mins, alpha, correlation_coefficient):
    n_vars = len(mins)

    # Create correlation matrix
    correlation_matrix = np.full((n_vars, n_vars), correlation_coefficient)
    np.fill_diagonal(correlation_matrix, 1)

    # add to diagonal to ensure positive definiteness
    correlation_matrix += np.eye(n_vars) * 1e-6

    # Generate correlated normal variables
    mean = np.zeros(n_vars)
    normal_samples = stats.multivariate_normal(mean, correlation_matrix).rvs(n_samples)

    # Transform to uniform using the standard normal CDF
    u = stats.norm.cdf(normal_samples)

    # Transform to Pareto using inverse CDF
    samples = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        samples[:, i] = mins[i] * (1 - u[:, i]) ** (-1 / alpha[i])

    return samples
