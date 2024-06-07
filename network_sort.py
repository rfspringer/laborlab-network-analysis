import networkx as nx
import utils


class NetworkSort:
    node_id_counter = 0
    graph = None

    def __init__(self, g: nx.DiGraph):
        self.graph = g.copy()
        assert utils.is_valid_network(self.graph)
        self.node_id_counter = max(self.graph.nodes()) + 1


    def sort(self):
        g = self.graph.copy()    # dont want to directly modify graph
        sorted = []

        g_sccs = self.collapse_SCCS(g)

        #sort strongly connected components by indegree
        sorted_sccs = self.sort_by_indegree(g_sccs)

        for i in range(len(sorted_sccs)):
            # sorted_sccs returns list of sets
            supernode_id_set = sorted_sccs[i]
            combined_subgraph = nx.DiGraph()

            # combined if supernodes are tied
            for id in supernode_id_set:
                combined_subgraph = nx.compose(combined_subgraph, g_sccs.nodes[id]['subnode'])

            sorted_subnodes = self.sort_by_net_flow(combined_subgraph)
            sorted = sorted + sorted_subnodes
        return sorted


    def calculate_net_flow(self, graph: nx.DiGraph, node):
        incoming_sum = sum(graph[u][node]['weight'] for u in graph.predecessors(node))  # note: this includes edges from outside the cycle
        outgoing_sum = sum(graph[node][v]['weight'] for v in graph.successors(node))
        return  outgoing_sum - incoming_sum


    def sort_by_indegree(self, graph: nx.DiGraph):
        g = graph.copy()    # dont want to directly modify graph
        assert nx.is_directed_acyclic_graph(g) # all supernodes should be collapsed

        # dictionary to map sorting keys to nodes
        # key is first by indegree, tiebreak with net flow
        # returns sets in case of tie
        sorting_keys = {}

        for node in g.nodes():
            topological_level = len(nx.ancestors(g, node))
            net_flow = self.calculate_net_flow(graph, node)    #TODO: should be static fn or in utils (same as topolgical order on)
            sorting_key = (topological_level, -net_flow)  # negate net flow to sort in descending order
            if sorting_key not in sorting_keys:
                sorting_keys[sorting_key] = set()  # initialize an empty set for the sorting key
            sorting_keys[sorting_key].add(node)  # add the node to the set

        # Sort nodes by indegree, tiebreak with net flow and net flow
        sorted_nodes = [value for key, value in sorted(sorting_keys.items())]

        return sorted_nodes
        # TODO: make sure to break ties w net flow


    def sort_by_net_flow(self, graph):  # TODO: should do this for all in same set, not just same SCC I think (in case two are at same position ex a-> b and a-> c, b and c are kind of tied
        net_flows = {}
        for node in graph.nodes():
            net_flow = self.calculate_net_flow(graph, node)
            if net_flow not in net_flows:
                net_flows[net_flow] = set()
            net_flows[net_flow].add(node)
        sorted_nodes = [net_flows[key] for key in sorted(net_flows.keys(), reverse=True)]
        return sorted_nodes


    def collapse_SCCS(self, g):
        sccs = list(nx.strongly_connected_components(g))

        for component in sccs:
            self.create_supernode(component, g)

        self_loops = [(u, v) for u, v in g.edges() if u == v]
        g.remove_edges_from(self_loops)
        return g


    def create_supernode(self, component, graph):
        supernode_id = self.node_id_counter
        self.node_id_counter += 1

        subgraph = self.graph.subgraph(component)   #use self here to make sure subcomponent retains the whole graph, despite changes
        graph.add_node(supernode_id, subnode= subgraph)

        for node in component:
            # aggregate edges for each node subcomponent
            for predecessor in graph.predecessors(node):
                weight = graph[predecessor][node]['weight']
                graph.add_edge(predecessor, supernode_id, weight=weight)

            for successor in graph.successors(node):
                weight = graph[node][successor]['weight']
                graph.add_edge(supernode_id, successor, weight=weight)

            graph.remove_node(node)


def test_sort():
    graph, order = utils.generate_weight_directed_cyclic()
    ns = NetworkSort(graph)
    test_graph(ns, order)


def test_graph(ns: NetworkSort, order: [set]):
    sorted = ns.sort(ns.graph)
    print(ns.graph)
    print("sorted: ", sorted)
    assert sorted == order, "Should be " + order


if __name__ == "__main__":
    test_sort()




