from tqdm import tqdm
import pickle as pkl
import json
import logging
from concurrent.futures import ProcessPoolExecutor
import os


def calculate_n_and_mean_labor_income(state_industry_dict, show_progress=True):
    total_labor_income = 0.
    n = 0

    iterable = state_industry_dict.items()
    if show_progress:
        iterable = tqdm(state_industry_dict.items(), desc="Calculating total labor income")

    for state_x_industry, records in iterable:
        for record in records:
            total_labor_income += record['labor_income']
            n += 1

    if n > 0:
        return n, total_labor_income / n
    else:
        return 0, None


def calculate_gini_denominator(state_industry_dict, show_progress=True):
    n, mean_labor_income = calculate_n_and_mean_labor_income(state_industry_dict, show_progress)
    return 2 * n * n * mean_labor_income


def calculate_exploitation_and_patronage_sums(state_industry_dict, show_progress=True):
    exploitation_sum = 0.
    patronage_sum = 0.

    total_pairs = sum(len(records_list) * (len(records_list) - 1) // 2 for records_list in state_industry_dict.values())
    progress_bar = tqdm(total=total_pairs, desc="Calculating exploitation and patronage sums") if show_progress else None

    for records_list in state_industry_dict.values():
        n = len(records_list)
        for i in range(n):
            record_1 = records_list[i]
            for j in range(i + 1, n):
                record_2 = records_list[j]
                domination_exists = record_1['capital_income'] != record_2['capital_income']
                if domination_exists:
                    higher_capital_income_has_higher_labor_income = \
                        (record_1['labor_income'] > record_2['labor_income'] and record_1['capital_income'] > record_2['capital_income']) or \
                        (record_2['labor_income'] > record_1['labor_income'] and record_2['capital_income'] > record_1['capital_income'])

                    labor_income_diff = abs(record_1['labor_income'] - record_2['labor_income'])
                    if higher_capital_income_has_higher_labor_income:
                        exploitation_sum += 2 * labor_income_diff
                    else:
                        patronage_sum += 2 * labor_income_diff

                if show_progress:
                    progress_bar.update(1)

    if show_progress:
        progress_bar.close()
    return exploitation_sum, patronage_sum


def calculate_exclusion_and_rationing_sums(state_industry_dict, show_progress=True):
    exclusion_sum = 0.
    rationing_sum = 0.

    total_comparisons = sum(
        len(records1) * len(records2) for i, records1 in enumerate(state_industry_dict.values()) for j, records2 in
        enumerate(state_industry_dict.values()) if i < j)
    progress_bar = tqdm(total=total_comparisons, desc="Calculating exclusion and rationing sums") if show_progress else None

    for i, records_list_1 in enumerate(state_industry_dict.values()):
        for record_1 in records_list_1:
            for j, records_list_2 in enumerate(state_industry_dict.values()):
                if i < j:  # Ensure only comparing records from different lists
                    for record_2 in records_list_2:
                        domination_exists = record_1['capital_income'] != record_2['capital_income']
                        if domination_exists:
                            higher_capital_income_has_higher_labor_income = \
                                (record_1['labor_income'] > record_2['labor_income'] and record_1['capital_income'] >
                                    record_2['capital_income']) or \
                                (record_2['labor_income'] > record_1['labor_income'] and record_2['capital_income'] >
                                    record_1['capital_income'])

                            labor_income_diff = abs(record_1['labor_income'] - record_2['labor_income'])
                            if higher_capital_income_has_higher_labor_income:
                                exclusion_sum += 2 * labor_income_diff
                            else:
                                rationing_sum += 2 * labor_income_diff

                        if show_progress:
                            progress_bar.update(1)

    if show_progress:
        progress_bar.close()
    return exclusion_sum, rationing_sum


def calculate_ginis(state_industry_dict, show_progress=True):
    denominator = calculate_gini_denominator(state_industry_dict, show_progress)
    exploitation_sum, patronage_sum = calculate_exploitation_and_patronage_sums(state_industry_dict, show_progress)
    exclusion_sum, rationing_sum = calculate_exclusion_and_rationing_sums(state_industry_dict, show_progress)

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


def filter_dict(state_industry_dict, min_age=None, max_age=None, filter_full_time_full_year=False, filter_bottom_99=False):
    filtered_state_industry_dict = {}

    for state_x_industry, records in state_industry_dict.items():
        filtered_records = []
        for record in records:
            if min_age is not None and record['AGE'] < min_age:
                continue
            if max_age is not None and record['AGE'] > max_age:
                continue
            if filter_full_time_full_year and not record.get('full_time_full_year', False):
                continue
            if filter_bottom_99 and not record.get('bottom_99', False):
                continue
            filtered_records.append(record)

        filtered_state_industry_dict[state_x_industry] = filtered_records

    return filtered_state_industry_dict


def calculate_all_ginis(min_age=None, max_age=None, filter_full_time_full_year=False, filter_bottom_99=False):
    min_year = 1975
    max_year = 2022
    results_dict = {}
    for year in range(min_year, max_year + 1):
        print(f"Calculating gini for year {year}")
        file_path = f'./data/cps_data/{year}_dict.pkl'
        with open(file_path, 'rb') as file:
            state_industry_dict = pkl.load(file)
        state_industry_dict = filter_dict(state_industry_dict, min_age, max_age, filter_full_time_full_year, filter_bottom_99)
        gini_dict = calculate_ginis(state_industry_dict)
        results_dict[year] = gini_dict
        print(results_dict)

    file_path = './results/gini_components_by_year.json'
    with open(file_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)


def calculate_gini_for_year(year, min_age=None, max_age=None, filter_full_time_full_year=False, filter_bottom_99=False, show_progress=False):
    file_path = f'./data/cps_data/{year}_dict.pkl'
    with open(file_path, 'rb') as file:
        state_industry_dict = pkl.load(file)
    state_industry_dict = filter_dict(state_industry_dict, min_age, max_age, filter_full_time_full_year, filter_bottom_99)
    gini_dict = calculate_ginis(state_industry_dict, show_progress)
    return year, gini_dict


def calculate_all_ginis_parallel(min_age=None, max_age=None, filter_full_time_full_year=False, filter_bottom_99=False):
    min_year = 1975
    max_year = 2022
    results_dict = {}

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_gini_for_year, year, min_age, max_age, filter_full_time_full_year, filter_bottom_99, False)
            for year in range(min_year, max_year + 1)
        ]
        for future in futures:
            year, gini_dict = future.result()
            results_dict[year] = gini_dict
            logging.info(f"Results for {year}: {gini_dict}")

    file_path = './results/gini_components_by_year.json'
    with open(file_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)

    logging.info(f"Data has been written to {file_path}")


if __name__ == "__main__":
    calculate_all_ginis_parallel(min_age=25, max_age=64, filter_full_time_full_year=True, filter_bottom_99=True)
