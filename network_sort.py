import networkx as nx
import utils


def _calculate_net_flow(graph: nx.DiGraph, node) -> float:
    """
    Calculate the net flow (weight of edges out - edges in) for the given node in the graph
    :param graph: graph or subgraph to calculate net flow in
    :param node: node to calculate
    :return: net flow for node in the graph
    """
    incoming_sum = sum(graph[u][node]["weight"] for u in graph.predecessors(node))
    outgoing_sum = sum(graph[node][v]["weight"] for v in graph.successors(node))
    return outgoing_sum - incoming_sum


def _sort_by_indegree(graph: nx.DiGraph) -> [set]:
    """
    Sorts an acyclic graph by indegree, with ties broken by net flow
    :param graph: Directed acyclic graph
    :return: list of sets of nodes, ordered first by topological order then by net flow
    """
    assert nx.is_directed_acyclic_graph(graph)  # all supernodes should be collapsed

    # dictionary of topological level and net flow as key and set of all nodes with those values
    # stored this way in case of ties
    sorting_keys = {}

    for node in graph.nodes():
        topological_level = len(nx.ancestors(graph, node))
        net_flow = _calculate_net_flow(graph, node)
        sorting_key = (
            topological_level,
            -net_flow,  # negate net flow to sort in descending order
        )
        if sorting_key not in sorting_keys:
            sorting_keys[sorting_key] = set()
        sorting_keys[sorting_key].add(node)

    # list of sets of nodes, sorted first by topological level, then by net flow
    sorted_nodes = [value for key, value in sorted(sorting_keys.items())]

    return sorted_nodes


def _sort_by_net_flow(graph: nx.DiGraph):
    """
    Sort nodes in the provided graph by netf low
    :param graph: weighted directed graph
    :return:  list of nodes, sorted by net flow (weight out - weight in)
    """
    net_flows = {}
    for node in graph.nodes():
        net_flow = _calculate_net_flow(graph, node)
        if net_flow not in net_flows:
            net_flows[net_flow] = set()  # stored in sets to allow for ties
        net_flows[net_flow].add(node)
    sorted_nodes = [net_flows[key] for key in sorted(net_flows.keys(), reverse=True)]
    return sorted_nodes


def _collapse_SCCS(g: nx.DiGraph, node_id_counter: int) -> (nx.DiGraph, int):
    """
    Collapses strongly connected components of a graph to supernodes
    :param g: Graph to collapse
    :param node_id_counter: local counter variable to keep track of node ids
    :return: Collapsed representation of graph, counter for node ids
    """
    sccs = list(nx.strongly_connected_components(g))

    for component in sccs:
        node_id_counter = _create_supernode(component, g, node_id_counter)

    self_loops = [(u, v) for u, v in g.edges() if u == v]
    g.remove_edges_from(self_loops)
    return g, node_id_counter


def _create_supernode(component: set, graph: nx.DiGraph, node_id_counter: int) -> int:
    """
    Collapses indicated component of graph into a supernode, maintaining edges in/ out
    :param component: set of nodes to collapse into supernode
    :param graph: graph to modify
    :param node_id_counter: local counter variable to keep track of node ids
    :return: updated node id counter
    """
    supernode_id = node_id_counter
    node_id_counter += 1

    # make subgraph of the indicated component and store in the subnode field of the supernode
    subgraph = graph.subgraph(component).copy()
    graph.add_node(
        supernode_id, subnode=subgraph
    )  # subnode stored the collapsed component

    for node in component:
        # maintain predecessors for collapsed nodes in supernode
        for predecessor in graph.predecessors(node):
            weight = graph[predecessor][node]["weight"]
            graph.add_edge(
                predecessor, supernode_id, weight=weight
            )  # note: adding to existing edges will cumulatively add weight

        # maintain successors for collapsed nodes in supernode
        for successor in graph.successors(node):
            weight = graph[node][successor]["weight"]
            graph.add_edge(supernode_id, successor, weight=weight)

        graph.remove_node(node)
    return node_id_counter


def sort(g: nx.DiGraph) -> [set]:
    """
    Sort directed graph by collapsing strongly connected components (SCCs) into supernodes,
    sorting supernodes by topological order and net flow,
    and then sorting collapsed components internally by net flow
    :param g: directed graph to sort
    :return: ordered list of sets of nodes
    """
    assert utils.is_valid_network(g)

    # if given graph is not weighted, add an equal weight to every edge
    if not nx.is_weighted(g):
        for u, v in g.edges():
            g[u][v]['weight'] = 1

    node_id_counter = (
        max(g.nodes()) + 1
    )  # node id counter to keep track of what id to give new nodes

    sorted_list = []

    # collapse graph into strongly connected components and sort supernodes
    g_sccs, node_id_counter = _collapse_SCCS(g.copy(), node_id_counter)
    sorted_sccs = _sort_by_indegree(g_sccs)

    # sort nodes within each strongly connected component internally, and place into ordered list
    for supernode_id_set in sorted_sccs:
        combined_subgraph = nx.DiGraph()

        for id in supernode_id_set:
            combined_subgraph = nx.compose(
                combined_subgraph, g_sccs.nodes[id]["subnode"]
            )

        sorted_subnodes = _sort_by_net_flow(combined_subgraph)
        sorted_list.extend(sorted_subnodes)

    return sorted_list


def test_sort():
    for test in utils.test_cases:
        graph, order = test()
        test_graph(graph, order)


def test_graph(graph: nx.DiGraph, order: [set]):
    sorted = sort(graph)
    print(nx.graph)
    print("sorted: ", sorted)
    assert sorted == order, "Should be " + order
    print("passed :)\n")


if __name__ == "__main__":
    test_sort()
