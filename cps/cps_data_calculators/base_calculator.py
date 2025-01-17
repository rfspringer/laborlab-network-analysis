import os
import pandas as pd
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed


class BaseCalculator(ABC):
    statefip_to_label = {
        1: 'Alabama',
        2: 'Alaska',
        4: 'Arizona',
        5: 'Arkansas',
        6: 'California',
        8: 'Colorado',
        9: 'Connecticut',
        10: 'Delaware',
        11: 'District of Columbia',
        12: 'Florida',
        13: 'Georgia',
        15: 'Hawaii',
        16: 'Idaho',
        17: 'Illinois',
        18: 'Indiana',
        19: 'Iowa',
        20: 'Kansas',
        21: 'Kentucky',
        22: 'Louisiana',
        23: 'Maine',
        24: 'Maryland',
        25: 'Massachusetts',
        26: 'Michigan',
        27: 'Minnesota',
        28: 'Mississippi',
        29: 'Missouri',
        30: 'Montana',
        31: 'Nebraska',
        32: 'Nevada',
        33: 'New Hampshire',
        34: 'New Jersey',
        35: 'New Mexico',
        36: 'New York',
        37: 'North Carolina',
        38: 'North Dakota',
        39: 'Ohio',
        40: 'Oklahoma',
        41: 'Oregon',
        42: 'Pennsylvania',
        44: 'Rhode Island',
        45: 'South Carolina',
        46: 'South Dakota',
        47: 'Tennessee',
        48: 'Texas',
        49: 'Utah',
        50: 'Vermont',
        51: 'Virginia',
        53: 'Washington',
        54: 'West Virginia',
        55: 'Wisconsin',
        56: 'Wyoming',
        61: 'Maine-New Hampshire-Vermont',
        65: 'Montana-Idaho-Wyoming',
        68: 'Alaska-Hawaii',
        69: 'Nebraska-North Dakota-South Dakota',
        70: 'Maine-Massachusetts-New Hampshire-Rhode Island-Vermont',
        71: 'Michigan-Wisconsin',
        72: 'Minnesota-Iowa',
        73: 'Nebraska-North Dakota-South Dakota-Kansas',
        74: 'Delaware-Virginia',
        75: 'North Carolina-South Carolina',
        76: 'Alabama-Mississippi',
        77: 'Arkansas-Oklahoma',
        78: 'Arizona-New Mexico-Colorado',
        79: 'Idaho-Wyoming-Utah-Montana-Nevada',
        80: 'Alaska-Washington-Hawaii',
        81: 'New Hampshire-Maine-Vermont-Rhode Island',
        83: 'South Carolina-Georgia',
        84: 'Kentucky-Tennessee',
        85: 'Arkansas-Louisiana-Oklahoma',
        87: 'Iowa-N Dakota-S Dakota-Nebraska-Kansas-Minnesota-Missouri',
        88: 'Washington-Oregon-Alaska-Hawaii',
        89: 'Montana-Wyoming-Colorado-New Mexico-Utah-Nevada-Arizona-Idaho',
        90: 'Delaware-Maryland-Virginia-West Virginia',
        99: 'State not identified'
    }

    def __init__(self, results_filepath):
        self.results_filepath = results_filepath

    def initialize_results_file(self, run_from_existing_file):
        """Initialize the results CSV file if it does not exist."""
        if not os.path.exists(self.results_filepath) or not run_from_existing_file:
            # Create an empty dataframe with expected columns from the base class or subclass
            df = pd.DataFrame(columns=self.get_result_columns())
            df.to_csv(self.results_filepath, index=False)

    def load_existing_results(self):
        """Load existing results from CSV and track processed year-state combinations."""
        if os.path.exists(self.results_filepath):
            df = pd.read_csv(self.results_filepath)
            # Track processed year-state combinations
            processed = df.groupby('Year', group_keys=False)['STATEFIP'].apply(set).to_dict()
            return df, processed
        return pd.DataFrame(), {}

    def filter_df(self, df, min_age=None, max_age=None, filter_full_time_full_year=False,
                  filter_top_1_percent=False, filter_ag_and_public_service=False, filter_by_sex=None):
        """Filter DataFrame based on criteria."""
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
        """Save results to CSV, appending new results."""
        if self.results_filepath:
            df = pd.DataFrame(results)
            columns = self.get_result_columns()
            df = df[columns]    # make sure column order matches up
            df.to_csv(self.results_filepath, mode='a', header=False, index=False)


    @abstractmethod
    def get_result_columns(self):
        """Return the list of column names for the results."""
        pass

    @abstractmethod
    def calculate_for(self, df, **kwargs):
        """Calculate results based on the filtered DataFrame."""
        pass

    def calculate_for_year_by_states(self, year_df, processed, **kwargs):
        """Calculate results for each state in a given year, skipping already processed states."""
        results = []
        states = set(year_df['STATEFIP'])

        for state in states:
            # Skip already processed states for this year
            if state in processed.get(year_df['YEAR'].iloc[0], set()):
                logging.info(
                    f"Skipping state {self.statefip_to_label[state]} ({state}) for year {year_df['YEAR'].iloc[0]} "
                    "because it has already been processed.")
                continue

            logging.info(
                f"Processing state {self.statefip_to_label[state]} ({state}) for year {year_df['YEAR'].iloc[0]}")
            state_year_df = year_df[year_df['STATEFIP'] == state]
            state_year_dict = self.calculate_for(state_year_df, **kwargs)

            state_year_results = {'Year': year_df['YEAR'].iloc[0], 'State': self.statefip_to_label[state],
                                  'STATEFIP': state}
            state_year_results.update(state_year_dict)
            results.append(state_year_results)

        return results  # Return the list of results for each state

    def calculate_for_year(self, year, calculate_by_states, processed, **kwargs):
        """Calculate results for the given year."""
        logging.info(f"Starting calculations for year {year}")
        results = []
        raw_year_df = pd.read_csv(f'../data/cps_data/{year}_sample.csv')

        # filter df
        valid_filter_args = ['min_age', 'max_age', 'filter_full_time_full_year',
                             'filter_top_1_percent', 'filter_ag_and_public_service', 'filter_by_sex']
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_filter_args}
        year_df = self.filter_df(raw_year_df, **filtered_kwargs)

        # Dynamically get states in the dataset for this year
        states_in_data = set(year_df['STATEFIP'])

        # Skip if all states in data have been processed
        if year in processed and states_in_data.issubset(processed.get(year, set())):
            logging.info(f"Skipping year {year} because all states in the data have already been processed.")
            return pd.DataFrame()

        # Calculate for the year (without state filter)
        logging.info(f"Calculating overall results for year {year}")
        results_dict = self.calculate_for(year_df, **kwargs)
        results_dict['Year'] = year
        results_dict['STATEFIP'] = None  # This will be None for the overall year calculation
        results_dict['State'] = None  # This will be None for the overall year calculation
        results.append(results_dict)

        # Calculate for states if needed
        if calculate_by_states:
            state_results = self.calculate_for_year_by_states(year_df, processed, **kwargs)
            results.extend(state_results)

        logging.info(f"Finished calculations for year {year}")
        return pd.DataFrame(results)  # Return a DataFrame for both state and non-state calculations

    def process_year(self, year, calculate_by_states, processed, **kwargs):
        """Process and calculate results for a single year."""
        results_df = self.calculate_for_year(year, calculate_by_states, processed, **kwargs)
        return results_df

    def calculate_all(self, min_year=1983, max_year=2023, parallel=False, run_from_existing_file=True,
                      calculate_by_states=False, **kwargs):
        """Calculate results for all years within the specified range."""
        self.initialize_results_file(run_from_existing_file)

        # Load existing results to determine which years and states to process
        existing_results, processed = self.load_existing_results()
        years_to_process = [year for year in range(min_year, max_year + 1)]

        logging.info(f"Starting calculations for years {min_year}-{max_year}")

        if parallel:
            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.process_year, year, calculate_by_states, processed, **kwargs): year for
                           year in years_to_process}

                with tqdm(total=len(years_to_process), desc="Processing all years", position=0) as pbar:
                    for future in as_completed(futures):
                        results_df = future.result()
                        if not results_df.empty:
                            self.save_results(results_df.to_dict('records'))
                        pbar.update(1)
        else:
            with tqdm(total=len(years_to_process), desc="Processing all years", position=0) as pbar:
                for year in years_to_process:
                    results_df = self.process_year(year, calculate_by_states, processed, **kwargs)
                    if not results_df.empty:
                        self.save_results(results_df.to_dict('records'))
                    pbar.update(1)

        logging.info(f"All calculations are complete. Data has been written to {self.results_filepath}")