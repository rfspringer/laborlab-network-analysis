from networkx import DiGraph
from network_utils import network_sort

def _generate_uniform_weight_directed_acyclic():
    G = DiGraph()
    G.add_edge(1, 2, weight=1)
    G.add_edge(1, 3, weight=1)
    G.add_edge(2, 4, weight=1)
    G.add_edge(2, 5, weight=1)
    G.add_edge(2, 6, weight=1)
    return G, [{1}, {2}, {3}, {4, 5, 6}]

def _generate_weight_directed_cyclic():
    G = DiGraph()
    G.add_edge(1, 2, weight=4)
    G.add_edge(2, 3, weight=1)
    G.add_edge(3, 1, weight=1)
    G.add_edge(2, 4, weight=15)
    return G, [{1}, {3}, {2}, {4}]

def _generate_unweighted_directed_cyclic():
    G = DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 1)
    G.add_edge(2, 4)
    G.add_edge(5, 4)
    return G, [{1, 2, 3, 5}, {4}]

test_cases = [
    _generate_uniform_weight_directed_acyclic,
    _generate_weight_directed_cyclic,
    _generate_unweighted_directed_cyclic
]

def test_sort():
    for test in test_cases:
        graph, order = test()
        sorted_result = network_sort.sort(graph)
        assert sorted_result == order, f"Failed for test {test.__name__}. Expected {order}, got {sorted_result}"
