import pytest
import networkx as nx
import utils

import network_sort
from graph_analysis import *

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
    average_diff = 3.8  # Calculated manually
    average_wealth = 6.5
    expected_gini = average_diff / (2 * average_wealth)
    connected_gini = wealth_gini_directly_connected_only(simple_test_graph)
    assert round(connected_gini, 3) == round(expected_gini, 3)


def test_wealth_gini_not_directly_connected (simple_test_graph):
    average_diff = 4.2  # Calculated manually
    average_wealth = 6.5
    expected_gini = average_diff / (2 * average_wealth)
    connected_gini = wealth_gini_not_directly_connected(simple_test_graph)
    assert round(connected_gini, 3) == round(expected_gini, 3)


def test_wealth_gini_weakly_connected(simple_test_graph):
    average_diff = 4.142856  # Calculated manually
    average_wealth = 6.5
    expected_gini = average_diff / (2 * average_wealth)
    connected_gini = wealth_gini_weakly_connected_only(simple_test_graph)
    assert round(connected_gini, 3) == round(expected_gini, 3)


def test_wealth_gini_not_weakly_connected(simple_test_graph):
    average_diff = 4.0  # Calculated manually
    average_wealth = 6.5
    expected_gini = average_diff / (2 * average_wealth)
    unconnected_gini = wealth_gini_weakly_unconnected_only(simple_test_graph)
    assert round(unconnected_gini, 3) == round(expected_gini, 3)


def test_wealth_gini_directly_connected_split_by_income(simple_test_graph):
    average_higher_income_diff = 4.5  # Calculated manually
    average_lower_income_diff = 1.0  # Calculated manually

    average_higher_income_wealth = 7.
    average_lower_income_wealth = 4.5
    expected_higher_income_gini = average_higher_income_diff / (2 * average_higher_income_wealth)
    expected_lower_income_gini = average_lower_income_diff / (2 * average_lower_income_wealth)

    higher_income_gini, lower_income_gini = wealth_gini_directly_connected_split_by_income(simple_test_graph)

    assert round(higher_income_gini, 3) == round(expected_higher_income_gini, 3)
    assert round(lower_income_gini, 3) == round(expected_lower_income_gini, 3)


def test_rank_correlation(simple_test_graph):
    sorted_nodes = network_sort.sort(simple_test_graph)

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

