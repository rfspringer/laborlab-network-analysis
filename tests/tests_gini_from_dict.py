import math

from gini_calculator_from_dict import *
import numpy as np
import math

# test_dict = {
#     'A': [{'labor_income': 5, 'capital_income': 10},
#           {'labor_income': 6, 'capital_income': 2},
#           {'labor_income': 5, 'capital_income': 15},
#           {'labor_income': 2, 'capital_income': 0}],
#     'B': [{'labor_income': 5, 'capital_income': 0},
#           {'labor_income': 10, 'capital_income': 15}, ],
#     'C': [{'labor_income': 1, 'capital_income': 5}]
# }
#
# test_dict = {k: np.array([(r['labor_income'], r['capital_income'], r.get('age', None),
#                                      r.get('full_time_full_year', False), r.get('bottom_99', True))
#                                     for r in v])
#                        for k, v in test_dict.items()}

test_df = pd.DataFrame([
    {'stateXindustry': 'A', 'labor_income': 5, 'capital_income': 10},
    {'stateXindustry': 'A', 'labor_income': 6, 'capital_income': 2},
    {'stateXindustry': 'A', 'labor_income': 5, 'capital_income': 15},
    {'stateXindustry': 'A', 'labor_income': 2, 'capital_income': 0},
    {'stateXindustry': 'B', 'labor_income': 5, 'capital_income': 0},
    {'stateXindustry': 'B', 'labor_income': 10, 'capital_income': 15},
    {'stateXindustry': 'C', 'labor_income': 1, 'capital_income': 5}
])


# Calculate correct_denominator once
correct_denominator = 2 * 7 * 7 * 4.85714285714

def test_denominator_calculation():
    assert math.isclose(calculate_gini_denominator(test_df), correct_denominator, abs_tol=10 ** -3)

def test_gini_exploitation():
    exploitation_sum = 2 * ((3 + 4 + 3) + (5))
    correct_exploitation_gini = exploitation_sum / correct_denominator

    calculated_ginis = calculate_ginis(test_df)
    assert math.isclose(correct_exploitation_gini, calculated_ginis['exploitation'], abs_tol=10 ** -3)


def test_gini_patronage():
    patronage_sum = 2 * (1 + 1)
    correct_patronage_gini = patronage_sum / correct_denominator

    calculated_ginis = calculate_ginis(test_df)
    assert math.isclose(correct_patronage_gini, calculated_ginis['patronage'], abs_tol=10 ** -3)


def test_gini_exclusion():
    exclusion_sum = 2 * (5 + 4 + 1 + 4 + 4 + 8 + 9)
    correct_exclusion_gini = exclusion_sum / correct_denominator

    calculated_ginis = calculate_ginis(test_df)
    assert math.isclose(correct_exclusion_gini, calculated_ginis['exclusion'], abs_tol=10 ** -3)


def test_gini_rationing():
    rationing_sum = 2 * (5 + 1 + 4)
    correct_rationing_gini = rationing_sum / correct_denominator

    calculated_ginis = calculate_ginis(test_df)
    assert math.isclose(correct_rationing_gini, calculated_ginis['rationing'], abs_tol=10 ** -3)


def test_gini_total():
    exploitation_sum = 2 * ((3 + 4 + 3) + (5))
    patronage_sum = 2 * (1 + 1)
    exclusion_sum = 2 * (5 + 4 + 1 + 4 + 4 + 8 + 9)
    rationing_sum = 2 * (5 + 1 + 4)
    correct_total_gini = (exploitation_sum + patronage_sum + exclusion_sum + rationing_sum) / correct_denominator

    calculated_ginis = calculate_ginis(test_df)
    assert math.isclose(correct_total_gini, calculated_ginis['total'], abs_tol=10 ** -3)


def test_gini_against_other_algorithm():
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
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    wealths = [1., 5., 72., 3., 6., 2.]
    graph_array = np.array(wealths)


    single_complete_network_dict = {'id': []}

    for i, val in enumerate(wealths):
        record = {'labor_income': val, 'capital_income': val}
        single_complete_network_dict['id'].append(record)

    single_complete_network_dict_np = {k: np.array([(r['labor_income'], r['capital_income'], r.get('age', None),
                                         r.get('full_time_full_year', False), r.get('bottom_99', True))
                                        for r in v])
                           for k, v in single_complete_network_dict.items()}

    gini_dict = calculate_ginis(single_complete_network_dict_np)

    assert math.isclose(gini_dict['total'], gini(graph_array), abs_tol=10 ** -3)
    assert math.isclose(gini_dict['exploitation'] + gini_dict['patronage'], gini(graph_array), abs_tol=10 ** -3)
    assert gini_dict['rationing'] == 0
    assert gini_dict['exclusion'] == 0

