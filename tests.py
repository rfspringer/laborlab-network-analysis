import pytest
import networkx as nx
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
    expected_gini = 4.07 # Calculated manually
    gini = wealth_gini_all_nodes(simple_test_graph)
    assert round(gini, 2) == expected_gini

def test_wealth_gini_directly_connected_only(simple_test_graph):
    expected_connected_gini = 3.8  # Calculated manually
    connected_gini = wealth_gini_directly_connected_only(simple_test_graph)
    assert round(connected_gini, 2) == expected_connected_gini

def test_wealth_gini_not_directly_connected (simple_test_graph):
    expected_connected_gini = 4.2  # Calculated manually
    connected_gini = wealth_gini_not_directly_connected(simple_test_graph)
    assert round(connected_gini, 2) == expected_connected_gini

def test_wealth_gini_weakly_connected(simple_test_graph):
    expected_connected_gini = 4.14  # Calculated manually
    connected_gini = wealth_gini_weakly_connected_only(simple_test_graph)
    assert round(connected_gini, 2) == expected_connected_gini

def test_wealth_gini_not_weakly_connected(simple_test_graph):
    expected_unconnected_gini = 4.0  # Calculated manually
    unconnected_gini = wealth_gini_weakly_unconnected_only(simple_test_graph)
    assert round(unconnected_gini, 2) == expected_unconnected_gini

def test_wealth_gini_directly_connected_split_by_income(simple_test_graph):
    expected_higher_income_gini = 4.5  # Calculated manually
    expected_lower_income_gini = 1.0  # Calculated manually

    higher_income_gini, lower_income_gini = wealth_gini_directly_connected_split_by_income(simple_test_graph)

    assert round(higher_income_gini, 2) == expected_higher_income_gini
    assert round(lower_income_gini, 2) == expected_lower_income_gini
