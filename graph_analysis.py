import networkx as nx
from itertools import combinations
from tqdm import tqdm


def get_wealth(node_data, wealth_attr):
    wealth = node_data.get(wealth_attr, 0)
    return wealth if wealth >= 0 else 0.


def get_diff(g: nx.DiGraph, wealth_attr, show_progress=True) -> (float, int):
    total_diff = 0.
    pair_count = 0

    node_pairs = list(combinations(g.nodes(), 2))
    iterator = tqdm(node_pairs, desc="Processing pairs") if show_progress else node_pairs

    for u, v in iterator:
        wealth_u = get_wealth(g.nodes[u], wealth_attr)
        wealth_v = get_wealth(g.nodes[v], wealth_attr)
        total_diff += abs(wealth_u - wealth_v)
        pair_count += 1

    return total_diff, pair_count


def get_mean_wealth(g: nx.DiGraph, wealth_attr) -> float:
    wealth_values = [get_wealth(data, wealth_attr) for _, data in g.nodes(data=True)]

    if wealth_values:
        average_wealth = sum(wealth_values) / len(wealth_values)
    else:
        average_wealth = None
    print(average_wealth)
    return average_wealth


def calculate_gini(average_diff, mean_wealth):
    return average_diff / (2 * mean_wealth)


def wealth_gini_all_nodes(g: nx.DiGraph, wealth_attr='wealth', mean_wealth=None) -> float:
    print("Calculating Gini for all node pairs...", flush=True)
    diff, pair_count = get_diff(g, wealth_attr)
    mean_wealth = mean_wealth if mean_wealth is not None else get_mean_wealth(g, wealth_attr)

    if mean_wealth is None or pair_count == 0:
        print("No pairs in graph, div by zero error", flush=True)
        return None  # Return None to indicate an undefined Gini coefficient

    average_diff = (diff / pair_count)
    gini = calculate_gini(average_diff, mean_wealth)
    print(f"Gini coefficient for all node pairs: {gini}")
    return gini


def wealth_gini_weakly_connected_only(g: nx.DiGraph, wealth_attr='wealth', mean_wealth=None) -> float:
    print("Calculating Gini for weakly connected components...", flush=True)
    components = list(nx.weakly_connected_components(g))
    total_diff = 0.
    num_pairs = 0

    for component in tqdm(components, desc="Processing components"):
        subgraph = g.subgraph(component)
        diff, pair_count = get_diff(subgraph, wealth_attr, show_progress=False)
        total_diff += diff
        num_pairs += pair_count

    if num_pairs == 0:
        return None  # No pairs to process

    average_diff = (total_diff / num_pairs)
    mean_wealth = mean_wealth if mean_wealth is not None else get_mean_wealth(g, wealth_attr)
    print(average_diff)
    print(mean_wealth)
    gini = calculate_gini(average_diff, mean_wealth)
    print(f"Gini coefficient for all node pairs: {gini}")
    return gini


def wealth_gini_directly_connected_only(g: nx.DiGraph, wealth_attr='wealth', mean_wealth=None) -> float:
    if mean_wealth is None:
        mean_wealth = get_mean_wealth(g, wealth_attr)


    print("Calculating Gini for directly connected node pairs...", flush=True)
    total_diff = 0.
    num_pairs = g.number_of_edges()

    for node_1, node_2 in tqdm(g.edges(), total=num_pairs, desc="Processing edges"):
        wealth_1 = get_wealth(g.nodes[node_1], wealth_attr)
        wealth_2 = get_wealth(g.nodes[node_2], wealth_attr)
        total_diff += abs(wealth_1 - wealth_2)

    if num_pairs == 0 or mean_wealth is None:
        return None  # No pairs to process

    average_diff = total_diff / num_pairs
    gini = calculate_gini(average_diff, mean_wealth)
    print(f"Gini coefficient for directly connected node pairs: {gini}")
    return gini


def wealth_gini_weakly_unconnected_only(g: nx.DiGraph, wealth_attr='wealth', mean_wealth=None) -> float:
    if mean_wealth is None:
        mean_wealth = get_mean_wealth(g, wealth_attr)

    if mean_wealth is None:
        return None

    print("Calculating Gini for weakly unconnected node pairs...", flush=True)
    total_diff = 0.
    num_pairs = 0
    connected_components = list(nx.weakly_connected_components(g))

    for i, component_1 in enumerate(tqdm(connected_components, desc="Processing components")):
        for component_2 in connected_components[i + 1:]:
            for node_1 in component_1:
                for node_2 in component_2:
                    wealth_1 = get_wealth(g.nodes[node_1], wealth_attr)
                    wealth_2 = get_wealth(g.nodes[node_2], wealth_attr)
                    total_diff += abs(wealth_1 - wealth_2)
                    num_pairs += 1

    if num_pairs == 0:
        return None  # No pairs to process

    average_diff = total_diff / num_pairs
    gini = calculate_gini(average_diff, mean_wealth)
    print(f"Gini coefficient for weakly unconnected node pairs: {gini}")
    return gini


def wealth_gini_not_directly_connected(g: nx.DiGraph, wealth_attr='wealth', mean_wealth=None) -> float:
    if mean_wealth is None:
        mean_wealth = get_mean_wealth(g, wealth_attr)

    if mean_wealth is None:
        return None

    print("Calculating Gini for not directly connected node pairs...", flush=True)
    total_diff = 0
    total_combinations = len(g.nodes) * (len(g.nodes) - 1) // 2
    num_pairs = 0

    for node_1, node_2 in tqdm(combinations(g.nodes(), 2), total=total_combinations, desc="Processing pairs"):
        if not g.has_edge(node_1, node_2):
            wealth_1 = get_wealth(g.nodes[node_1], wealth_attr)
            wealth_2 = get_wealth(g.nodes[node_2], wealth_attr)
            total_diff += abs(wealth_1 - wealth_2)
            num_pairs += 1

    if num_pairs == 0:
        return None  # No pairs to process

    average_diff = total_diff / num_pairs
    gini = calculate_gini(average_diff, mean_wealth)
    print(f"Gini coefficient for not directly connected node pairs: {gini}")
    return gini


def wealth_gini_directly_connected_split_by_income(g: nx.DiGraph, wealth_attr='wealth',
                                                   income_attr='income', mean_wealth=None) -> (float, float):
    if mean_wealth is None:
        mean_wealth = get_mean_wealth(g, wealth_attr)

    if mean_wealth is None:
        return None, None

    print("Calculating Gini for directly connected node pairs split by income...", flush=True)

    wealthier_higher_income_diff = 0.
    wealthier_lower_income_diff = 0.
    wealthier_higher_num_pairs = 0
    wealthier_lower_num_pairs = 0

    total_edges = g.number_of_edges()
    for node_1, node_2 in tqdm(g.edges(), total=total_edges, desc="Processing edges"):
        wealth_1 = get_wealth(g.nodes[node_1], wealth_attr)
        wealth_2 = get_wealth(g.nodes[node_2], wealth_attr)
        diff = abs(wealth_1 - wealth_2)

        higher_wealth_node = node_1 if wealth_1 >= wealth_2 else node_2
        lower_wealth_node = node_2 if higher_wealth_node == node_1 else node_1
        higher_income_node = higher_wealth_node if get_wealth(g.nodes[higher_wealth_node], income_attr) >= get_wealth(
            g.nodes[lower_wealth_node], income_attr) else lower_wealth_node

        if higher_wealth_node == higher_income_node:
            wealthier_higher_income_diff += diff
            wealthier_higher_num_pairs += 1
        else:
            wealthier_lower_income_diff += diff
            wealthier_lower_num_pairs += 1

    wealthier_higher_average_diff = (
                wealthier_higher_income_diff / wealthier_higher_num_pairs) if wealthier_higher_num_pairs != 0 else 0.
    wealthier_lower_average_diff = (
                wealthier_lower_income_diff / wealthier_lower_num_pairs) if wealthier_lower_num_pairs != 0 else 0.

    wealthier_higher_income_gini = calculate_gini(wealthier_higher_average_diff, mean_wealth)
    wealthier_lower_income_gini = calculate_gini(wealthier_lower_average_diff, mean_wealth)

    print(f"Gini coefficient for wealthier-higher-income pairs: {wealthier_higher_income_gini}")
    print(f"Gini coefficient for wealthier-lower-income pairs: {wealthier_lower_income_gini}")

    return wealthier_higher_income_gini, wealthier_lower_income_gini


def calculate_all_ginis(g: nx.DiGraph, wealth_attr='wealth', income_attr='income', mean_wealth=None):
    if mean_wealth is None:
        mean_wealth = get_mean_wealth(g, wealth_attr)

    if mean_wealth is None:
        return None

    print(f"Mean wealth: {mean_wealth}")

    ginis = {
        "all_nodes": wealth_gini_all_nodes(g, wealth_attr, mean_wealth),
        "weakly_connected_only": wealth_gini_weakly_connected_only(g, wealth_attr, mean_wealth),
        "directly_connected_only": wealth_gini_directly_connected_only(g, wealth_attr, mean_wealth),
        "weakly_unconnected_only": wealth_gini_weakly_unconnected_only(g, wealth_attr, mean_wealth),
        "not_directly_connected": wealth_gini_not_directly_connected(g, wealth_attr, mean_wealth),
        "directly_connected_split_by_income": wealth_gini_directly_connected_split_by_income(g, wealth_attr, income_attr, mean_wealth)
    }

    return ginis
