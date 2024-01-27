import networkx as nx
import utils
import matplotlib


def sort(graph: nx.DiGraph):
    g = graph.copy()    # dont want to directly modify graph
    assert utils.is_valid_network(g)
    sorted = []
    while g.number_of_nodes() > 0:
        nodes = get_nodes_with_indegree_zero(g)
        sorted.append(nodes)
        g.remove_nodes_from(nodes)
    return sorted


def get_nodes_with_indegree_zero(graph: nx.DiGraph):
    node_set = set()
    for node, in_degree in graph.in_degree:
        if in_degree == 0:
            node_set.add(node)
    return node_set


def test_sort():
    graph, order = utils.generate_uniform_weight_directed_acyclic()
    test_graph(graph, order)


def test_graph(graph, order: [set]):
    sorted = sort(graph)
    print(graph)
    print("sorted: ", sorted)
    assert sort(graph) == order, "Should be " + order


if __name__ == "__main__":
    test_sort()


# TODO:
# take in network x graph

# make sample graph ( simple, unweighted, directed acyclic)
# topo sort- throw error if cyclic

# make sample graph (simple, unweighted, directed but with cycle)
# topo sort- collapse cyclic into pseudonodes

# make sample graph (simple, weighted, directed but with cycle)
# make topo sort with weights


# topo sort with weights- but keep cyclic collapsed

# toposort to unpack cyclic

# return ordered list of set of nodes at each level



