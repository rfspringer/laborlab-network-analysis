import networkx as nx
import utils
import matplotlib


class NetworkSort:
    node_id_counter = 0
    graph = None

    # all nodes have attribute: id, is_supernode
    # have global current_id tracker
    # supernode contains attr of cycle within it (to sort)
    def __init__(self, g: nx.DiGraph):
        self.graph = g.copy()
        assert utils.is_valid_network(self.graph)
        nx.set_node_attributes(self.graph, {node: False for node in self.graph.nodes}, "is_supernode")  # set is_supernode to false for all nodes
        self.node_id_counter = max(self.graph.nodes()) + 1


    def sort(self, graph):
        g = graph.copy()    # dont want to directly modify graph
        # assert contains is_supernode attrs?

        g_collapsed, is_single_cycle = self.collapse_cycles()

        if is_single_cycle:
            sorted = self.sort_cycle(g)
        else:
            sorted = self.sort_by_indegree(g_collapsed)

        for i in range(len(sorted)):
            node = sorted[i]
            # recursively sort pseudonodes
            if g_collapsed.nodes[node].get("is_supernode"):
                subgraph = g_collapsed.nodes[node].get("subgraph")
                sorted_subgraph = self.sort(subgraph)
                sorted = sorted[:i] + sorted_subgraph + sorted[i+1:]    # splice in supernode element positions
        return sorted


    def sort_cycle(self, graph):
        net_flows = {}
        for node in graph.nodes():
            incoming_sum = sum(graph[u][node]['weight'] for u in graph.predecessors(node))
            outgoing_sum = sum(graph[node][v]['weight'] for v in graph.successors(node))
            net_flows[node] = outgoing_sum - incoming_sum

        sorted_nodes = sorted(net_flows, key=net_flows.get, reverse=True)   # sort nodes by net flows
        return sorted_nodes


    def sort_by_indegree(self, graph: nx.DiGraph):
        g = graph.copy()    # dont want to directly modify graph
        assert nx.is_directed_acyclic_graph(g) # all pseudonodes should be collapsed
        sorted = []

        while g.number_of_nodes() > 0:
            nodes = self.get_nodes_with_indegree_zero(g)
            sorted.append(nodes)
            g.remove_nodes_from(nodes)
        return sorted


    def collapse_cycles(self):
        g = self.graph.copy()
        basis_cycles = utils.get_basis(g)
        if len(basis_cycles) == 1:
            # dont collapse cycle spanning entire graph
            return g, True

        for cycle in basis_cycles:
            self.create_pseudonode(cycle, g)
        return g, False


    def create_pseudonode(self, cycle, graph):
        # add node w feature is_supernode = true, and id of the next id
        # if cycle is not a cycle, then make sure not a supernode
        supernode_id = self.node_id_counter
        self.node_id_counter += 1

        subgraph = graph.subgraph(cycle)
        graph.add_node(supernode_id, {'is_supernode': True, 'graph': subgraph})

        for node in cycle:
            neighbors = list(graph.neighbors(node))
            graph.remove_node(node)
            graph.add_edges_from([(supernode_id, neighbor) for neighbor in neighbors])


    def get_nodes_with_indegree_zero(self, graph: nx.DiGraph):
        node_set = set()
        for node, in_degree in graph.in_degree:
            if in_degree == 0:
                node_set.add(node)
        return node_set



def test_sort():
    graph, order = utils.generate_uniform_weight_directed_acyclic()
    ns = NetworkSort(graph)
    test_graph(ns, order)


def test_graph(ns: NetworkSort, order: [set]):
    sorted = ns.sort(ns.graph)
    print(ns.graph)
    print("sorted: ", sorted)
    assert sorted == order, "Should be " + order


if __name__ == "__main__":
    test_sort()


    # node objects:
    # contain id, contents
    # contents are node or value

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



