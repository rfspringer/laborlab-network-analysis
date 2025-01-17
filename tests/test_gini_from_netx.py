import pytest
import networkx as nx
from network_utils.gini_from_netx import GiniCalculatorFromNetX, DominationStrategies

@pytest.fixture
def single_test_graph():
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

@pytest.fixture
def income_only_test_graph():
    g = nx.DiGraph()
    g.add_nodes_from([
        ('A', {'income': 5}),
        ('B', {'income': 3}),
        ('C', {'income': 2}),
        ('D', {'income': 3}),
        ('E', {'income': 3}),
        ('F', {'income': 1}),
    ])
    g.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('E', 'F')
    ])
    return g

@pytest.fixture
def wealth_only_test_graph():
    g = nx.DiGraph()
    g.add_nodes_from([
        ('A', {'wealth': 10}),
        ('B', {'wealth': 8}),
        ('C', {'wealth': 5}),
        ('D', {'wealth': 4}),
        ('E', {'wealth': 10}),
        ('F', {'wealth': 2}),
    ])
    g.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('E', 'F')
    ])
    return g

@pytest.fixture
def edge_test_graph():
    g = nx.DiGraph()
    g.add_nodes_from([
        ('A', {'wealth': 10}),
        ('B', {'wealth': 8}),
        ('C', {'wealth': 5}),
        ('D', {'wealth': 4}),
        ('E', {'wealth': 10}),
        ('F', {'wealth': 2}),
    ])

    # easy rule for manually calculating
    g.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('A', 'F'), ('B', 'C'), ('B', 'D'), ('B', 'E'), ('B', 'F'), ('C', 'D'), ('C', 'E'), ('C', 'F'), ('D', 'E'), ('D', 'F'), ('E', 'F')
    ])
    return g

correct_denominator = 2 * 6 * (5 + 3 + 2 + 3 + 3 + 1)


def test_gini_outputs_same_with_two_graphs(single_test_graph, income_only_test_graph, wealth_only_test_graph):
    gini_calculator_1 = GiniCalculatorFromNetX(income_graph=single_test_graph)
    results_1 = gini_calculator_1.calculate_components(DominationStrategies.attribute_comparison, domination_attr='wealth')

    gini_calculator_2 = GiniCalculatorFromNetX(income_graph=single_test_graph, domination_graph=single_test_graph)
    results_2 = gini_calculator_2.calculate_components(DominationStrategies.attribute_comparison, domination_attr='wealth')

    assert results_1 == results_2

def test_gini_outputs_correct_with_attr_comparison(single_test_graph, income_only_test_graph, wealth_only_test_graph):
    #from single graph
    gini_calculator_1 = GiniCalculatorFromNetX(income_graph=single_test_graph)
    results_1 = gini_calculator_1.calculate_components(DominationStrategies.attribute_comparison, domination_attr='wealth')

    # from two graphs
    gini_calculator_2 = GiniCalculatorFromNetX(income_graph=single_test_graph, domination_graph=single_test_graph)
    results_2 = gini_calculator_2.calculate_components(DominationStrategies.attribute_comparison, domination_attr='wealth')

    correct_results = {
        'exploitation': (2 + 3 + 1 + 2) * 2 / correct_denominator,
        'patronage': (1) * 2 / correct_denominator,
        'exclusion': (2 + 4 + 2 + 2 + 2) * 2 / correct_denominator,
        'rationing': 0.,
        'component_sum':(2 + 3 + 1 + 2 + 1 + 2 + 4 + 2 + 2 + 2) * 2 / correct_denominator,
        'total_gini': (2 + 3 + 1 + 2 + 1 + 2 + 4 + 2 + 2 + 2 + 2) * 2 / correct_denominator
    }

    assert results_1 == correct_results
    assert results_2 == correct_results

def test_gini_outputs_correct_with_edge_comparison(income_only_test_graph, edge_test_graph):
    gini_calculator = GiniCalculatorFromNetX(income_graph=income_only_test_graph, domination_graph=edge_test_graph)
    results = gini_calculator.calculate_components(DominationStrategies.directed_edge)

    correct_results = {
        'exploitation': (2 + 3 + 1 + 2) * 2 / correct_denominator,
        'patronage': (1) * 2 / correct_denominator,
        'exclusion': (2 + 2 + 4 + 2 + 1 + 2) * 2 / correct_denominator,
        'rationing': (1) * 2 / correct_denominator,
        'component_sum':(2 + 3 + 1 +  2 + 1 + 2 + 2 + 4 + 2 + 1 + 2 + 1) * 2 / correct_denominator,
        'total_gini': (2 + 3 + 1 +  2 + 1 + 2 + 2 + 4 + 2 + 1 + 2 + 1) * 2 / correct_denominator
    }

    assert results == correct_results

def test_correct_denominator(single_test_graph):
    gini_calculator = GiniCalculatorFromNetX(single_test_graph)
    assert correct_denominator == gini_calculator.denominator