import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

class BaseCalculator(ABC):
    def __init__(self, results_filepath):
        self.results_filepath = results_filepath

    def initialize_results_file(self):
        if not os.path.exists(self.results_filepath):
            with open(self.results_filepath, 'w') as json_file:
                json.dump({}, json_file)

    def load_existing_results(self):
        if os.path.exists(self.results_filepath):
            with open(self.results_filepath, 'r') as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return {}
        return {}

    def get_years_to_process(self, min_year, max_year, run_from_existing_results):
        if run_from_existing_results:
            print("Loading existing results from same filename")
            results_dict = self.load_existing_results()
            if results_dict:
                result_years = [int(year) for year in results_dict.keys()]
                return [year for year in range(min_year, max_year + 1) if year not in result_years]
        
        return list(range(min_year, max_year + 1))

    def filter_df(self, df, min_age=None, max_age=None, filter_full_time_full_year=False,
                 filter_top_1_percent=False, filter_ag_and_public_service=False, filter_by_sex=None):
        mask = pd.Series(True, index=df.index)

        if min_age is not None:
            mask &= (df['AGE'] >= min_age)
        if max_age is not None:
            mask &= (df['AGE'] <= max_age)
        if filter_full_time_full_year:
            mask &= df['is_full_time_full_year'].astype(bool)
        if filter_top_1_percent:
            mask &= df['winsor99'].astype(bool)
        if filter_ag_and_public_service:
            ind1950_codes = df['IND1950'].astype(str).str[0].astype(int)
            mask &= ~ind1950_codes.isin([1, 9])
        if filter_by_sex == "MEN":
            mask &= (df['SEX'] == 1)
        elif filter_by_sex == "WOMEN":
            mask &= (df['SEX'] == 2)

        return df[mask]

    def save_results(self, results):
        if self.results_filepath:
            try:
                with open(self.results_filepath, 'r+') as json_file:
                    try:
                        json_data = json.load(json_file)
                    except json.JSONDecodeError:
                        json_data = {}
                    json_data.update(results)
                    json_file.seek(0)
                    json.dump(json_data, json_file, indent=4)
                    json_file.truncate()
            except FileNotFoundError:
                with open(self.results_filepath, 'w') as json_file:
                    json.dump(results, json_file, indent=4)

    @abstractmethod
    def calculate_for_year(self, year, **kwargs):
        pass

    def process_year(self, year, **kwargs):
        year, results = self.calculate_for_year(year, **kwargs)
        print(f"\n\nResults for {year}: {results}\n\n")
        return year, results

    def calculate_all(self, min_year=1983, max_year=2022, parallel=False, run_from_existing_file=True, **kwargs):
        self.initialize_results_file()
        years_to_process = self.get_years_to_process(min_year, max_year, run_from_existing_file)

        if parallel:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(self.process_year, year, **kwargs): year
                    for year in years_to_process
                }

                with tqdm(total=len(years_to_process), desc="Processing all years", position=0) as pbar:
                    for future in as_completed(futures):
                        year, results = future.result()
                        self.save_results({year: results})
                        pbar.update(1)
        else:
            with tqdm(total=len(years_to_process), desc="Processing all years", position=0) as pbar:
                for year in years_to_process:
                    year, results = self.process_year(year, **kwargs)
                    self.save_results({year: results})
                    pbar.update(1)

        logging.info(f"Data has been written to {self.results_filepath}")
