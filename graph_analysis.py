import networkx as nx
from itertools import combinations, product
from tqdm import tqdm


def get_wealth(node_data, wealth_attr):
    wealth = node_data.get(wealth_attr, 0)
    return wealth if wealth > 0 else 0.0001


def get_mean_wealth_of_nodes(g, nodes, wealth_attr):
    wealth_values = [get_wealth(g.nodes[n], wealth_attr) for n in nodes]
    if wealth_values:
        return sum(wealth_values) / len(wealth_values)
    else:
        return None


def calculate_gini(average_diff, mean_wealth):
    if mean_wealth == 0:
        return None
    else:
        return average_diff / (2 * mean_wealth)


def get_all_pairs(g: nx.DiGraph):
    return [(u, v) for u, v in combinations(g.nodes(), 2)]


def get_weakly_connected_pairs(g: nx.DiGraph):
    pairs = []
    for component in nx.weakly_connected_components(g):
        for u, v in combinations(component, 2):
            pairs.append((u, v))
    return pairs


def get_not_weakly_connected_pairs(g: nx.DiGraph):
    weakly_connected_components = list(nx.weakly_connected_components(g))
    pairs = []
    for i in range(len(weakly_connected_components)):
        for j in range(i + 1, len(weakly_connected_components)):
            pairs.extend(product(weakly_connected_components[i], weakly_connected_components[j]))
    return pairs


def get_directly_connected_pairs(g: nx.DiGraph):
    return list(g.edges())


def get_not_directly_connected_pairs(g: nx.DiGraph):
    nodes = list(g.nodes())
    all_pairs = combinations(nodes, 2)
    not_directly_connected_pairs = [
        (u, v) for u, v in all_pairs if not g.has_edge(u, v)
    ]
    return not_directly_connected_pairs


def get_directly_connected_pairs_by_wealth_and_income(g: nx.DiGraph, wealth_attr='wealth', income_attr='income'):
    higher_wealth_higher_income = []
    higher_wealth_lower_income = []

    for u, v in g.edges():
        wealth_u = g.nodes[u].get(wealth_attr, 0)
        wealth_v = g.nodes[v].get(wealth_attr, 0)
        income_u = g.nodes[u].get(income_attr, 0)
        income_v = g.nodes[v].get(income_attr, 0)

        if (wealth_u > wealth_v and income_u >= income_v) or (wealth_v > wealth_u and income_v >= income_u):
            higher_wealth_higher_income.append((u, v))
        else:
            higher_wealth_lower_income.append((u, v))

    return higher_wealth_higher_income, higher_wealth_lower_income


def wealth_gini_from_pairs(g: nx.DiGraph, pairs, wealth_attr='wealth') -> float:
    total_diff = 0.0
    wealth_sum = 0.0
    total_pairs = len(pairs)

    for u, v in tqdm(pairs, total=total_pairs, desc="Processing node pairs"):
        wealth_u = get_wealth(g.nodes[u], wealth_attr)
        wealth_v = get_wealth(g.nodes[v], wealth_attr)
        total_diff += abs(wealth_u - wealth_v)
        wealth_sum += (wealth_u + wealth_v)

    print("Calculating Gini Coefficient...")
    average_diff = total_diff / total_pairs
    average_wealth = wealth_sum / (2 * total_pairs)
    gini = calculate_gini(average_diff, average_wealth)

    return gini


def wealth_gini_all_nodes(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating gini for all nodes...")
    pairs = get_all_pairs(g)
    gini = wealth_gini_from_pairs(g, pairs, wealth_attr)
    print(f"Gini coefficient for all node pairs: {gini}")
    return gini


def wealth_gini_weakly_connected(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating gini for weakly connected nodes...")
    pairs = get_weakly_connected_pairs(g)
    gini = wealth_gini_from_pairs(g, pairs, wealth_attr)
    print(f"Gini coefficient for weakly connected node pairs: {gini}")
    return gini


def wealth_gini_not_weakly_connected(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating gini for not weakly connected nodes...")
    pairs = get_not_weakly_connected_pairs(g)
    gini = wealth_gini_from_pairs(g, pairs, wealth_attr)
    print(f"Gini coefficient for not weakly connected node pairs: {gini}")
    return gini


def wealth_gini_directly_connected(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating gini for directly connected nodes...")
    pairs = get_directly_connected_pairs(g)
    gini = wealth_gini_from_pairs(g, pairs, wealth_attr)
    print(f"Gini coefficient for directly connected node pairs: {gini}")
    return gini


def wealth_gini_not_directly_connected(g: nx.DiGraph, wealth_attr='wealth') -> float:
    print("Calculating gini for not directly connected nodes...")
    pairs = get_not_directly_connected_pairs(g)
    gini = wealth_gini_from_pairs(g, pairs, wealth_attr)
    print(f"Gini coefficient for not directly connected node pairs: {gini}")
    return gini


def wealth_gini_directly_connected_split_by_income(g: nx.DiGraph, wealth_attr='wealth', income_attr='income') -> (float, float):
    print("Calculating Gini for directly connected node pairs split by income...", flush=True)
    higher_income_pairs, lower_income_pairs = \
        get_directly_connected_pairs_by_wealth_and_income(g, wealth_attr, income_attr)
    higher_income_gini = wealth_gini_from_pairs(g, higher_income_pairs, wealth_attr)
    lower_income_gini = wealth_gini_from_pairs(g, lower_income_pairs, wealth_attr)

    print(f"Gini coefficient for wealthier-higher-income pairs: {higher_income_gini}")
    print(f"Gini coefficient for wealthier-lower-income pairs: {lower_income_gini}")

    return higher_income_gini, lower_income_gini


def calculate_all_ginis(g: nx.DiGraph, wealth_attr='wealth', income_attr='income'):
    ginis = {
        "all_nodes": wealth_gini_all_nodes(g, wealth_attr),
        "weakly_connected_only": wealth_gini_weakly_connected(g, wealth_attr),
        "not_weakly_connected_only": wealth_gini_not_weakly_connected(g, wealth_attr),
        "not_directly_connected": wealth_gini_not_directly_connected(g, wealth_attr),
        "directly_connected_split_by_income": wealth_gini_directly_connected_split_by_income(g, wealth_attr, income_attr)
    }

    return ginis