import os
import numpy as np
import pickle as pkl
import json
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from threading import Lock
import pandas as pd

# Global lock for file writing
file_lock = Lock()


def calculate_n_and_mean_labor_income(df):
    n = len(df)
    mean_labor_income = df['labor_income'].mean()
    return n, mean_labor_income


def calculate_gini_denominator(df):
    n, mean_labor_income = calculate_n_and_mean_labor_income(df)
    return 2 * n * n * mean_labor_income


def calculate_exploitation_and_patronage_sums(df, use_1950_industry_codes=False, outer_tqdm=None):
    exploitation_sum = 0.
    patronage_sum = 0.

    industry_column = 'stateXindustry_1950' if use_1950_industry_codes else 'stateXindustry'

    # Group by specified industry column
    for state_x_industry, group in tqdm(df.groupby(industry_column), desc="Calculating exploitation and patronage",
                                        leave=False, position=outer_tqdm):
        # Calculate pairwise differences for labor and capital income (bidirectional, so no *2 needed)
        labor_diff_matrix = group['labor_income'].values[:, None] - group['labor_income'].values
        capital_diff_matrix = group['capital_income'].values[:, None] - group['capital_income'].values

        # Domination mask and higher capital and labor correlation
        domination_mask = capital_diff_matrix != 0
        higher_capital_higher_labor = (labor_diff_matrix * capital_diff_matrix > 0)

        # Calculate exploitation and patronage sums
        exploitation_sum += np.sum(np.abs(labor_diff_matrix) * (domination_mask & higher_capital_higher_labor))
        patronage_sum += np.sum(np.abs(labor_diff_matrix) * (domination_mask & ~higher_capital_higher_labor))

    return exploitation_sum, patronage_sum


def calculate_exclusion_and_rationing_sums(df, use_1950_industry_codes=False, outer_tqdm=None):
    exclusion_sum = 0.
    rationing_sum = 0.

    industry_column = 'stateXindustry_1950' if use_1950_industry_codes else 'stateXindustry'

    # Group data by specified industry column and get the list of groups
    state_x_industry_groups = list(df.groupby(industry_column))

    # Prepare arrays of labor and capital income
    group_labor_incomes = [group['labor_income'].values for _, group in state_x_industry_groups]
    group_capital_incomes = [group['capital_income'].values for _, group in state_x_industry_groups]

    # Calculate the total number of pairs
    total_pairs = len(state_x_industry_groups) * (len(state_x_industry_groups) - 1) // 2

    # Initialize tqdm for the pairs of groups
    with tqdm(total=total_pairs, desc="Calculating exclusion and rationing", leave=False, position=outer_tqdm.position+1 if outer_tqdm else 1) as pbar:
        # Iterate over pairs of groups using a triangular loop to avoid double counting
        for i in range(len(group_labor_incomes)):
            group1_labor = group_labor_incomes[i]
            group1_capital = group_capital_incomes[i]
            for j in range(i + 1, len(state_x_industry_groups)):
                group2_labor = group_labor_incomes[j]
                group2_capital = group_capital_incomes[j]

                # Compute pairwise differences
                labor_diff = group1_labor[:, None] - group2_labor
                capital_diff = group1_capital[:, None] - group2_capital

                # Vectorized computation of exclusion and rationing
                domination_mask = capital_diff != 0
                higher_capital_higher_labor = (labor_diff * capital_diff > 0)

                exclusion_sum += np.sum(np.abs(labor_diff[domination_mask & higher_capital_higher_labor]))
                rationing_sum += np.sum(np.abs(labor_diff[domination_mask & ~higher_capital_higher_labor]))

                # Update the progress bar for each pair processed
                pbar.update(1)

    # Multiplied by two because counted unidirectionally
    return exclusion_sum * 2, rationing_sum * 2



def calculate_ginis(df, use_1950_industry_codes=False, outer_tqdm=None):
    denominator = calculate_gini_denominator(df)

    exploitation_sum, patronage_sum = calculate_exploitation_and_patronage_sums(df, use_1950_industry_codes, outer_tqdm)
    exclusion_sum, rationing_sum = calculate_exclusion_and_rationing_sums(df, use_1950_industry_codes, outer_tqdm)

    if denominator == 0:
        raise ValueError("Denominator for Gini calculation is zero. Check the input data.")

    gini_dict = {
        'exploitation': exploitation_sum / denominator,
        'patronage': patronage_sum / denominator,
        'exclusion': exclusion_sum / denominator,
        'rationing': rationing_sum / denominator,
        'total': (exploitation_sum + patronage_sum + exclusion_sum + rationing_sum) / denominator
    }

    return gini_dict


def filter_df(df, min_age=None, max_age=None, filter_full_time_full_year=False, filter_top_1_percent=False,
              outer_tqdm=None):
    mask = np.ones(len(df), dtype=bool)

    if min_age is not None:
        mask &= (df['AGE'] >= min_age)
    if max_age is not None:
        mask &= (df['AGE'] <= max_age)
    if filter_full_time_full_year:
        mask &= df['is_full_time_full_year'].astype(bool)
    if filter_top_1_percent:
        mask &= df['winsor99'].astype(bool)

    return df.loc[mask]


def calculate_gini_for_year(year, min_age=None, max_age=None, filter_full_time_full_year=False,
                            filter_top_1_percent=False, use_1950_industry_codes=False, outer_tqdm=None):
    year_df = pd.read_csv(f'./data/cps_data/{year}_sample.csv')
    year_df = filter_df(year_df, min_age, max_age, filter_full_time_full_year, filter_top_1_percent, outer_tqdm)
    gini_dict = calculate_ginis(year_df, use_1950_industry_codes, outer_tqdm)

    return year, gini_dict


def initialize_results_file(results_filepath):
    """ Initialize results file with an empty JSON dictionary if it does not exist. """
    if not os.path.exists(results_filepath):
        with open(results_filepath, 'w') as json_file:
            json.dump({}, json_file)


def load_existing_results(results_filepath):
    """ Load existing results from the file. """
    if os.path.exists(results_filepath):
        with open(results_filepath, 'r') as json_file:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
                return {}
    return {}


def get_years_to_process(results_dict, min_year, max_year):
    """ Get the next year to process based on existing results. """
    if results_dict:
        result_years = [int(year) for year in results_dict.keys()]
        years_not_in_results = [year for year in range(min_year, max_year + 1) if year not in result_years]
        return years_not_in_results
    return [year for year in range(min_year, max_year + 1)]


def process_year(year, min_age=None, max_age=None, filter_full_time_full_year=False, filter_top_1_percent=False,
                 use_1950_industry_codes=False, results_filepath=None):
    year, gini_dict = calculate_gini_for_year(year, min_age, max_age, filter_full_time_full_year,
                                              filter_top_1_percent, use_1950_industry_codes)

    print(f"\n\nresults for {year}: {gini_dict}\n\n")
    if results_filepath:
        # Append result to the JSON dictionary in the file
        with file_lock, open(results_filepath, 'r+') as json_file:
            try:
                json_data = json.load(json_file)
            except json.JSONDecodeError:
                json_data = {}  # Handle case where the file might be empty or corrupted

            json_data[str(year)] = gini_dict
            json_file.seek(0)
            json.dump(json_data, json_file, indent=4)
            json_file.write('\n')
            json_file.flush()

    return year, gini_dict


def calculate_all_ginis(min_age=None, max_age=None, filter_full_time_full_year=False, filter_top_1_percent=False,
                        results_filepath='./results/gini_components_by_year.json', min_year=1976, max_year=2022,
                        parallel=False, use_1950_industry_codes=False):
    # Initialize the results file
    initialize_results_file(results_filepath)

    # Determine the start year
    results_dict = load_existing_results(results_filepath)
    years_to_process = get_years_to_process(results_dict, min_year, max_year)
    print("processing: ", years_to_process)

    if parallel:
        # Prepare argument tuples for starmap
        arguments = [(year, min_age, max_age, filter_full_time_full_year, filter_top_1_percent,
                      use_1950_industry_codes, results_filepath) for year in years_to_process]

        with Pool(processes=2) as pool:
            with tqdm(total=len(years_to_process), desc="Calculating Gini for all years", position=0) as pbar:
                for _ in pool.starmap(process_year, arguments):
                    pbar.update(1)
    else:
        with tqdm(total=len(years_to_process), desc="Calculating Gini for all years", position=0) as pbar:
            for year in years_to_process:
                process_year(year, min_age, max_age, filter_full_time_full_year, filter_top_1_percent,
                             use_1950_industry_codes, results_filepath)
                pbar.update(1)

    logging.info(f"Data has been written to {results_filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting!")
    calculate_all_ginis(min_age=18, max_age=None, filter_full_time_full_year=False, filter_top_1_percent=False,
                        results_filepath='results/gini_components_by_year_18plus_newind.json', parallel=False,
                        use_1950_industry_codes=False)
    calculate_all_ginis(min_age=18, max_age=None, filter_full_time_full_year=False, filter_top_1_percent=True,
                        results_filepath='results/gini_components_by_year_18plus_winsorized_newind.json', parallel=False,
                        use_1950_industry_codes=False)
    # calculate_all_ginis(min_age=18, max_age=None, filter_full_time_full_year=False, filter_top_1_percent=False,
    #                     min_year=1994, results_filepath='results/gini_components_by_year_18plus_1994plus.json',
    #                     parallel=True, use_1950_industry_codes=True)
