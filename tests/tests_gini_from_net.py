import pytest
import numpy as np

import network_utils
from network_utils.gini_calculation_from_network import *

@pytest.fixture
def simple_test_graph():
    def create_test_graph():
        g = nx.DiGraph()
        g.add_nodes_from([
            ('A', {'wealth': 10, 'income': 5}),
            ('B', {'wealth': 8, 'income': 3}),
            ('C', {'wealth': 5, 'income': 2}),
            ('D', {'wealth': 4, 'income': 3}),
            ('E', {'wealth': 10, 'income': 3}),
            ('F', {'wealth': 2, 'income': 1}),
        ])
        g.add_edges_from([
            ('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('E', 'F')
        ])
        return g
    return create_test_graph()


def test_wealth_gini_all_nodes(simple_test_graph):
    average_diff = 4.066666666667
    average_wealth = 6.5
    expected_gini = average_diff / (2 * average_wealth)
    gini = wealth_gini_all_nodes(simple_test_graph)
    assert round(gini, 3) == round(expected_gini, 3)


def test_wealth_gini_directly_connected_only(simple_test_graph):
    expected_gini = 0.28358 # Calculated manually
    connected_gini = wealth_gini_directly_connected(simple_test_graph)
    assert round(connected_gini, 3) == round(expected_gini, 3)


def test_wealth_gini_not_directly_connected (simple_test_graph):
    expected_gini = 0.328125    # Calculated manually
    connected_gini = wealth_gini_not_directly_connected(simple_test_graph)
    assert round(connected_gini, 3) == round(expected_gini, 3)


def test_wealth_gini_weakly_connected(simple_test_graph):
    expected_gini = 0.31183 # Calculated manually
    connected_gini = wealth_gini_weakly_connected(simple_test_graph)
    assert round(connected_gini, 3) == round(expected_gini, 3)


def test_wealth_gini_not_weakly_connected(simple_test_graph):
    expected_gini = 0.31373  # Calculated manually
    unconnected_gini = wealth_gini_not_weakly_connected(simple_test_graph)
    assert round(unconnected_gini, 3) == round(expected_gini, 3)


def test_wealth_gini_directly_connected_split_by_income(simple_test_graph):
    expected_higher_income_gini = 0.31034
    expected_lower_income_gini = 0.11111

    higher_income_gini, lower_income_gini = wealth_gini_directly_connected_split_by_income(simple_test_graph)

    assert round(higher_income_gini, 3) == round(expected_higher_income_gini, 3)
    assert round(lower_income_gini, 3) == round(expected_lower_income_gini, 3)


def test_rank_correlation(simple_test_graph):
    sorted_nodes = network_utils.sort(simple_test_graph)

    # Sort nodes based on their 'wealth' attribute values
    wealths = nx.get_node_attributes(simple_test_graph, 'wealth')
    wealth_sorted_nodes = sorted(simple_test_graph.nodes(), key=lambda node: wealths[node], reverse=True)
    ranks = {id_: rank for rank, id_ in enumerate(wealth_sorted_nodes, start=1)}
    print(ranks)

    rho, list = utils.calculate_rank_correlation(sorted_nodes, ranks)
    print(sorted_nodes)
    print(wealth_sorted_nodes)
    print(list)
    print(rho)


def test_complete_graph_output():
    def gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        # All values are treated equally, arrays must be 1d:
        array = array.flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
        # Values cannot be 0:
        array += 0.0000001
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1, array.shape[0] + 1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))) * n/(n-1)

    wealths = [1., 5., 72., 3., 6., 2.]
    graph_array = np.array(wealths)

    complete_graph = nx.DiGraph()
    for i, weight in enumerate(wealths):
        complete_graph.add_node(i, wealth=weight)

    # Add directed edges
    for i in range(len(wealths)):
        for j in range(i + 1, len(wealths)):
            complete_graph.add_edge(i, j)

    assert round(gini(graph_array), 3) == round(wealth_gini_directly_connected(complete_graph), 3)
    assert round(gini(graph_array), 3) == round(wealth_gini_weakly_connected(complete_graph), 3)
    assert round(gini(graph_array), 3) == round(wealth_gini_all_nodes(complete_graph), 3)
