import networkx as nx
from itertools import combinations


def wealth_gini_all_nodes(g: nx.DiGraph) -> float:
    for node in g.nodes(data=True):
        assert 'wealth' in node[1], f"Node {node[0]} does not have a 'wealth' attribute"

    summed_difference, pairs = get_summed_diff_num_pairs(g)

    if pairs == 0:
        print("No pairs in graph, div by zero error")
        gini = None
    else:
        gini = summed_difference / pairs
    return gini

def get_summed_diff_num_pairs(g: nx.DiGraph) -> (float, int):
    pairs = 0
    summed_difference = 0
    for u, v in combinations(g.nodes(), 2):
        wealth_u = g.nodes[u]['wealth']
        wealth_v = g.nodes[v]['wealth']
        summed_difference += abs(wealth_u - wealth_v)
        pairs += 1
    return summed_difference, pairs


def wealth_gini_weakly_connected_only(g: nx.DiGraph) -> float:
    for node in g.nodes(data=True):
        assert 'wealth' in node[1], f"Node {node[0]} does not have a 'wealth' attribute"

    components = nx.weakly_connected_components(g)
    summed_diffs = 0
    num_pairs = 0
    for component in components:
        diff, pairs = get_summed_diff_num_pairs(g.subgraph(component))
        summed_diffs += diff
        num_pairs += pairs

    if num_pairs == 0:
        print("No weakly connected pairs in graph, div by zero error")
        gini = None
    else:
        gini = summed_diffs / num_pairs
    return gini


def wealth_gini_directly_connected_only(g: nx.DiGraph) -> float:
    for node in g.nodes(data=True):
        assert 'wealth' in node[1], f"Node {node[0]} does not have a 'wealth' attribute"

    num_pairs = g.number_of_edges()
    total_diff = 0
    for node_1, node_2 in g.edges():
        # Get the wealth of the two connected nodes
        wealth_1 = g.nodes[node_1]['wealth']
        wealth_2 = g.nodes[node_2]['wealth']

        diff = abs(wealth_1 - wealth_2)
        total_diff += diff

    if num_pairs == 0:
        print("No directly connected pairs in graph, div by zero error")
        gini = None
    else:
        gini = total_diff / num_pairs
    return gini


def wealth_gini_weakly_unconnected_only(g: nx.DiGraph) -> float:
    for node in g.nodes(data=True):
        assert 'wealth' in node[1], f"Node {node[0]} does not have a 'wealth' attribute"

    total_difference = 0
    pair_count = 0

    connected_components = nx.weakly_connected_components(g)
    for i, component_1 in enumerate(connected_components):
        for component_2 in connected_components[i + 1:]:
            for node_1 in component_1:
                for node_2 in component_2:
                    wealth_1 = g.nodes[node_1]['wealth']
                    wealth_2 = g.nodes[node_2]['wealth']
                    difference = abs(wealth_1 - wealth_2)
                    total_difference += difference
                    pair_count += 1

    if pair_count == 0:
        print("No weakly unconnected pairs in graph, div by zero error")
        gini = None
    else:
        gini = total_difference / pair_count
    return gini


def wealth_gini_not_directly_connected(g: nx.DiGraph) -> float:
    total_diff = 0
    unconnected_pairs = 0
    for node_1, node_2 in combinations(g.nodes(), 2):
        if not g.has_edge(node_1, node_2):
            wealth_1 = g.nodes[node_1]['wealth']
            wealth_2 = g.nodes[node_2]['wealth']

            diff = abs(wealth_1 - wealth_2)
            total_diff += diff
            unconnected_pairs += 1

    if unconnected_pairs == 0:
        print("No not directly connected pairs in graph, div by zero error")
        gini = None
    else:
        gini = total_diff / unconnected_pairs
    return gini


def wealth_gini_directly_connected_split_by_income(g: nx.DiGraph) -> (float, float):
    wealthier_higher_income_diffs = 0
    wealthier_lower_income_diffs = 0

    wealthier_higher_income_pairs = 0
    wealthier_lower_income_pairs = 0
    for node_1, node_2 in g.edges():
        wealth_1 = g.nodes[node_1]['wealth']
        wealth_2 = g.nodes[node_2]['wealth']
        diff = abs(wealth_1 - wealth_2)

        # or do this by direction of arrow?
        higher_wealth_node = node_1 if g.nodes[node_1].get('income') > g.nodes[node_1].get('income') else node_2
        higher_income_node = node_1 if g.nodes[node_1].get('wealth') > g.nodes[node_1].get('wealth') else node_2
        if higher_wealth_node is higher_income_node:
            wealthier_higher_income_diffs += diff
            wealthier_higher_income_pairs += 1
        else:
            wealthier_lower_income_diffs += diff
            wealthier_lower_income_pairs += 1

    higher_income_gini = wealthier_higher_income_diffs/wealthier_higher_income_pairs if wealthier_higher_income_pairs is not 0 else None
    lower_income_gini = wealthier_lower_income_diffs / wealthier_lower_income_pairs if wealthier_lower_income_pairs is not 0 else None
    return higher_income_gini, lower_income_gini


