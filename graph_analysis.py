import networkx as nx
from itertools import combinations

def wealth_gini_all_nodes(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating Gini for all node pairs...", flush=True)
    summed_difference, pairs = get_summed_diff_num_pairs(g, wealth_attr)

    if pairs == 0:
        print("No pairs in graph, div by zero error", flush=True)
        gini = None
    else:
        gini = summed_difference / pairs
    return gini

def get_summed_diff_num_pairs(g: nx.DiGraph, wealth_attr) -> (float, int):
    pairs = 0
    summed_difference = 0
    total_nodes = len(g.nodes)
    total_combinations = total_nodes * (total_nodes - 1) // 2

    for i, (u, v) in enumerate(combinations(g.nodes(), 2), 1):
        if i % 100000 == 0 or i == total_combinations:
            print(f"\rProcessed {i} out of {total_combinations} pairs...", end='', flush=True)
        wealth_u = g.nodes[u][wealth_attr]
        wealth_v = g.nodes[v][wealth_attr]
        summed_difference += abs(wealth_u - wealth_v)
        pairs += 1
    return summed_difference, pairs


def wealth_gini_weakly_connected_only(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating Gini for weakly connected components...", flush=True)
    components = list(nx.weakly_connected_components(g))
    summed_diffs = 0
    num_pairs = 0
    total_components = len(components)
    for i, component in enumerate(components, 1):
        print(f"\rProcessing component {i} out of {total_components}...", end='', flush=True)
        diff, pairs = get_summed_diff_num_pairs(g.subgraph(component), wealth_attr)
        summed_diffs += diff
        num_pairs += pairs


    if num_pairs == 0:
        print("No weakly connected pairs in graph, div by zero error", flush=True)
        gini = None
    else:
        gini = summed_diffs / num_pairs
    return gini


def wealth_gini_directly_connected_only(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating Gini for directly connected node pairs...", flush=True)
    num_pairs = g.number_of_edges()
    total_diff = 0
    for i, (node_1, node_2) in enumerate(g.edges(), 1):
        if i % 1000 == 0 or i == num_pairs:
            print(f"\rProcessed {i} out of {num_pairs} edges...", end='', flush=True)
        wealth_1 = g.nodes[node_1][wealth_attr]
        wealth_2 = g.nodes[node_2][wealth_attr]

        diff = abs(wealth_1 - wealth_2)
        total_diff += diff

    print(f"\rProcessed {num_pairs} out of {num_pairs} edges...", flush=True)
    if num_pairs == 0:
        print("No directly connected pairs in graph, div by zero error", flush=True)
        gini = None
    else:
        gini = total_diff / num_pairs
    return gini


def wealth_gini_weakly_unconnected_only(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating Gini for weakly unconnected node pairs...", flush=True)
    total_difference = 0
    pair_count = 0

    connected_components = list(nx.weakly_connected_components(g))
    total_components = len(connected_components)
    for i, component_1 in enumerate(connected_components):
        print(f"\rProcessing component {i + 1} out of {total_components}...", end='', flush=True)
        for component_2 in connected_components[i + 1:]:
            for node_1 in component_1:
                for node_2 in component_2:
                    wealth_1 = g.nodes[node_1][wealth_attr]
                    wealth_2 = g.nodes[node_2][wealth_attr]
                    difference = abs(wealth_1 - wealth_2)
                    total_difference += difference
                    pair_count += 1

    print(f"\rProcessed {total_components} out of {total_components} components...", flush=True)
    if pair_count == 0:
        print("No weakly unconnected pairs in graph, div by zero error", flush=True)
        gini = None
    else:
        gini = total_difference / pair_count
    return gini


def wealth_gini_not_directly_connected(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating Gini for not directly connected node pairs...", flush=True)
    total_diff = 0
    unconnected_pairs = 0
    total_nodes = len(g.nodes)
    total_combinations = total_nodes * (total_nodes - 1) // 2
    for i, (node_1, node_2) in enumerate(combinations(g.nodes(), 2), 1):
        if not g.has_edge(node_1, node_2):
            if i % 100000 == 0 or i == total_combinations:
                print(f"\rProcessed {i} out of {total_combinations} pairs...", end='', flush=True)
            wealth_1 = g.nodes[node_1][wealth_attr]
            wealth_2 = g.nodes[node_2][wealth_attr]

            diff = abs(wealth_1 - wealth_2)
            total_diff += diff
            unconnected_pairs += 1

    print(f"\rProcessed {total_combinations} out of {total_combinations} pairs...", flush=True)
    if unconnected_pairs == 0:
        print("No not directly connected pairs in graph, div by zero error", flush=True)
        gini = None
    else:
        gini = total_diff / unconnected_pairs
    return gini


def wealth_gini_directly_connected_split_by_income(g: nx.DiGraph, wealth_attr='wealth') -> (float, float):
    print("Calculating Gini for directly connected node pairs split by income...", flush=True)
    wealthier_higher_income_diffs = 0
    wealthier_lower_income_diffs = 0

    wealthier_higher_income_pairs = 0
    wealthier_lower_income_pairs = 0
    total_edges = g.number_of_edges()
    for i, (node_1, node_2) in enumerate(g.edges(), 1):
        if i % 1000 == 0 or i == total_edges:
            print(f"\rProcessed {i} out of {total_edges} edges...", end='', flush=True)
        wealth_1 = g.nodes[node_1][wealth_attr]
        wealth_2 = g.nodes[node_2][wealth_attr]
        diff = abs(wealth_1 - wealth_2)

        higher_wealth_node = node_1 if g.nodes[node_1].get('wealth') >= g.nodes[node_2].get('wealth') else node_2
        lower_wealth_node = node_2 if higher_wealth_node is node_1 else node_1
        higher_income_node = higher_wealth_node if g.nodes[higher_wealth_node].get('income') >= g.nodes[lower_wealth_node].get('income') else lower_wealth_node
        if higher_wealth_node is higher_income_node:
            wealthier_higher_income_diffs += diff
            wealthier_higher_income_pairs += 1
        else:
            wealthier_lower_income_diffs += diff
            wealthier_lower_income_pairs += 1

    print(f"\rProcessed {total_edges} out of {total_edges} edges...", flush=True)
    higher_income_gini = wealthier_higher_income_diffs / wealthier_higher_income_pairs if wealthier_higher_income_pairs != 0 else None
    lower_income_gini = wealthier_lower_income_diffs / wealthier_lower_income_pairs if wealthier_lower_income_pairs != 0 else None
    return higher_income_gini, lower_income_gini
