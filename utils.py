import networkx as nx
from itertools import permutations, product


def is_valid_network(graph: nx.DiGraph):
    if not isinstance(graph, nx.DiGraph):
        return False, "Must input NetworkX graph"
    elif not nx.is_directed(graph):
        return False, "Graph must be directed"
    else:
        return True


def generate_uniform_weight_directed_acyclic():
    # correct ordering: [{1}, {2, 3}, {4, 5, 6}]
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=1)
    G.add_edge(1, 3, weight=1)
    G.add_edge(2, 4, weight=1)
    G.add_edge(2, 5, weight=1)
    G.add_edge(2, 6, weight=1)
    return G, [{1}, {2}, {3}, {4, 5, 6}]


def generate_weight_directed_cyclic():
    # correct ordering: [{1}, {3}, {2}, {4}]
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=4)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 1, weight=1)
    G.add_edge(2, 4, weight=15)
    return G, [{1}, {3}, {2}, {4}]

def generate_unweighted_directed_cyclic():
    # correct ordering: [{1, 2, 3, 5}, {4}]
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 1)
    G.add_edge(2, 4)
    G. add_edge(5, 4)
    return G, [{1, 2, 3, 5}, {4}]

test_cases = [generate_uniform_weight_directed_acyclic, generate_weight_directed_cyclic, generate_unweighted_directed_cyclic]


# Function to check linear independence of cycles
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
