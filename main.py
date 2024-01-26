import networkx as nx
import utils


def sort(graph: nx.DiGraph):
    assert utils.is_valid_network(graph)

def test_sort():
    graph, order = utils.generate_uniform_weight_directed_acyclic()
    unweighted_acyclic = graph
    assert sort(graph) == order

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



