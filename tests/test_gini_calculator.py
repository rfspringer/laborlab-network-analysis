from cps.cps_data_calculators.gini_calculator import GiniCalculator
import numpy as np
import math
import pandas as pd

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
test_calculator = GiniCalculator(None)

def test_denominator_calculation():
    # GiniCalculator.calculate_gini_denominator(test_df)
    assert math.isclose(test_calculator.calculate_gini_denominator(test_df), correct_denominator, abs_tol=10 ** -3)

def test_gini_exploitation():
    exploitation_sum = 2 * ((3 + 4 + 3) + (5))
    correct_exploitation_gini = exploitation_sum / correct_denominator

    calculated_ginis = test_calculator.calculate_ginis(test_df, group_identifier='stateXindustry')
    assert math.isclose(correct_exploitation_gini, calculated_ginis['exploitation'], abs_tol=10 ** -3)


def test_gini_patronage():
    patronage_sum = 2 * (1 + 1)
    correct_patronage_gini = patronage_sum / correct_denominator

    calculated_ginis = test_calculator.calculate_ginis(test_df, group_identifier='stateXindustry')
    assert math.isclose(correct_patronage_gini, calculated_ginis['patronage'], abs_tol=10 ** -3)


def test_gini_exclusion():
    exclusion_sum = 2 * (5 + 4 + 1 + 4 + 4 + 8 + 9)
    correct_exclusion_gini = exclusion_sum / correct_denominator

    calculated_ginis = test_calculator.calculate_ginis(test_df, group_identifier='stateXindustry')
    assert math.isclose(correct_exclusion_gini, calculated_ginis['exclusion'], abs_tol=10 ** -3)


def test_gini_rationing():
    rationing_sum = 2 * (5 + 1 + 4)
    correct_rationing_gini = rationing_sum / correct_denominator

    calculated_ginis = test_calculator.calculate_ginis(test_df, group_identifier='stateXindustry')
    assert math.isclose(correct_rationing_gini, calculated_ginis['rationing'], abs_tol=10 ** -3)


def test_gini_total():
    exploitation_sum = 2 * ((3 + 4 + 3) + (5))
    patronage_sum = 2 * (1 + 1)
    exclusion_sum = 2 * (5 + 4 + 1 + 4 + 4 + 8 + 9)
    rationing_sum = 2 * (5 + 1 + 4)
    correct_total_gini = (exploitation_sum + patronage_sum + exclusion_sum + rationing_sum) / correct_denominator

    calculated_ginis = test_calculator.calculate_ginis(test_df, group_identifier='stateXindustry')
    assert math.isclose(correct_total_gini, calculated_ginis['total_gini'], abs_tol=10 ** -3)
    assert math.isclose(correct_total_gini, calculated_ginis['component_sum'], abs_tol=10 ** -3)


def test_gini_against_other_algorithm_single_net():
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

    # Convert to a pandas DataFrame
    single_complete_network_df = pd.DataFrame({
        'labor_income': wealths,
        'capital_income': wealths,
        'age': [None] * len(wealths),
        'full_time_full_year': [False] * len(wealths),
        'winsor99': [True] * len(wealths),
        'stateXindustry': 'test'
    })

    # Call the test calculator
    gini_dict = test_calculator.calculate_ginis(single_complete_network_df, group_identifier='stateXindustry')

    # Assertions
    assert math.isclose(gini_dict['total'], gini(graph_array), abs_tol=10 ** -3)
    assert math.isclose(gini_dict['exploitation'] + gini_dict['patronage'], gini(graph_array), abs_tol=10 ** -3)
    assert gini_dict['rationing'] == 0
    assert gini_dict['exclusion'] == 0


def test_gini_against_other_algorithm_two_nets():
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

    wealths_1 = [1., 5., 72.]
    wealths_2 = [3., 6., 2.]
    graph_array = np.array(wealths_1 + wealths_2)

    # Create DataFrames for each group
    df1 = pd.DataFrame({
        'labor_income': wealths_1,
        'capital_income': wealths_1,
        'age': [None] * len(wealths_1),
        'full_time_full_year': [False] * len(wealths_1),
        'winsor99': [True] * len(wealths_1),
        'stateXindustry': 'test1'
    })

    df2 = pd.DataFrame({
        'labor_income': wealths_2,
        'capital_income': wealths_2,
        'age': [None] * len(wealths_2),
        'full_time_full_year': [False] * len(wealths_2),
        'winsor99': [True] * len(wealths_2),
        'stateXindustry': 'test2'
    })

    # Combine the DataFrames
    network_df = pd.concat([df1, df2], ignore_index=True)

    # Call the test calculator
    gini_dict = test_calculator.calculate_ginis(network_df, group_identifier='stateXindustry')

    # Assertions
    assert math.isclose(gini_dict['total_gini'], gini(graph_array), abs_tol=10 ** -3)
    assert math.isclose(gini_dict['exploitation'] + gini_dict['patronage'] + gini_dict['rationing'] + gini_dict['exclusion'], gini(graph_array), abs_tol=10 ** -3)
    # assert math.isclose(gini_dict['exploitation'] + gini_dict['patronage'], gini(graph_array), abs_tol=10 ** -3)
    # assert gini_dict['rationing'] == 0
    # assert gini_dict['exclusion'] == 0

