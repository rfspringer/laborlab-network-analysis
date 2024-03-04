import networkx as nx


def is_valid_network(graph: nx.DiGraph):
    if not isinstance(graph, nx.DiGraph):
        return False, "Must input NetworkX graph"
    elif not nx.is_directed(graph):
        return False, "Graph must be directed"
    elif not nx.is_weighted(graph):
        return False, "Graph must have weights"
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
    return G, [{1}, {2, 3}, {4, 5, 6}]


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


