import pandas as pd
from ipumspy import readers
import numpy as np
import logging
import warnings


# File paths
cps_filepath = '../data/cps_data.dat.gz'
cps_codebook_filepath = '../data/cps_codebook.xml'
swap_filename = '../data/swapvalues_1976-2010.csv'

# Constants
INCOME_VARS = ['INCTOT', 'INCWAGE', 'INCBUS', 'INCFARM', 'INCINT', 'INCDRT', 'INCRENT', 'INCDIVID',
               'INCSS', 'INCWELFR', 'INCLONGJ', 'OINCBUS', 'OINCWAGE', 'OINCFARM', 'HHINCOME', 'INCGOV', 'INCALOTH', 'INCRETIR', 'INCSSI', 'INCUNEMP', 'INCOTHER']
CAPITAL_INCOME_VARS = ['INCDRT', 'INCINT']
LABOR_INCOME_VARS = ['INCWAGE', 'INCBUS', 'INCFARM']
VARS_TO_SET_NONNEG = ['INCRENT', 'INCDRT', 'INCBUS', 'INCFARM'] # rent and self-employment income variables
MIN_DATA_YEAR = 1976
YEAR_OF_SEPARATE_QUESTIONS_FOR_DIFFERENT_JOBS = 1988
HARD_MIN_AGE = 16

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_swap_data(swap_df):
    """Clean and filter swap data for relevant years."""
    swap_df = swap_df.fillna(0)
    swap_df.columns = swap_df.columns.str.upper()   # capitalize to match cps data columns
    return swap_df[swap_df['YEAR'] >= MIN_DATA_YEAR]


def change_income_dtypes_to_float(df):
    # otherwise theres issues multiplying them later
    for var in INCOME_VARS:
        df[var] = df[var].astype(float)
    return df


def fix_capital_income_topcoding(df):
    """Adjust specific topcoding inconsistencies in capital income data."""
    df['INCINT'] = df['INCINT'].replace(9999997., 99999.)
    df['INCRENT'] = df['INCRENT'].replace(9999997., 99999.)
    return df


def merge_cps_with_swap_data(cps_df, swap_df):
    """Merge CPS data with swap data."""
    merged = pd.merge(cps_df, swap_df, on=['YEAR', 'SERIAL', 'PERNUM'], how='left', indicator=True)
    merged['swap'] = merged['_merge'] == 'both'
    return merged.drop('_merge', axis=1)


def fill_in_topcodes(df):
    """Fill in topcoded values using swap data."""

    # starting in 1988, separate questions were introduced for earnings on the main job held in the previous year and other jobs
    topcode_columns = {
        'before_1988': ['INCWAGE', 'INCBUS', 'INCFARM'],
        'after_1987': ['INCLONGJ', 'OINCWAGE', 'OINCBUS', 'OINCFARM'],
        'all_years': ['INCINT', 'INCDRT', 'INCRENT', 'INCDIVID', 'INCSS', 'INCWELFR']
    }
    conditions = {
        'before_1988': (df['YEAR'] < YEAR_OF_SEPARATE_QUESTIONS_FOR_DIFFERENT_JOBS),
        'after_1987': (df['YEAR'] >= YEAR_OF_SEPARATE_QUESTIONS_FOR_DIFFERENT_JOBS),
        'all_years': pd.Series([True] * len(df))
    }
    for period, columns in topcode_columns.items():
        for col in columns:
            mask = (df[f'{col}_SWAP'].notna() & (df[f'{col}_SWAP'] != 0) & conditions[period])
            df.loc[mask, col] = df[f'{col}_SWAP']
    return df


def manually_fix_mistakes(df):
    """Manually fix known data mistakes."""
    df.loc[(df['YEAR'] == 1990) & (df['INCLONGJ_SWAP'] == 399998.), 'INCLONGJ_SWAP'] = 300000.
    return df

def reconstruct_wage_and_se_earnings_after_1988(df):
    """Reconstruct wage and self-employment earnings after 1988."""
    mask = (df['YEAR'] >= YEAR_OF_SEPARATE_QUESTIONS_FOR_DIFFERENT_JOBS)
    df.loc[mask & (df['SRCEARN'] == 1), 'INCWAGE'] = df['INCLONGJ'] + df['OINCWAGE']
    df.loc[mask & (df['SRCEARN'] == 2), 'INCBUS'] = df['INCLONGJ'] + df['OINCBUS']
    df.loc[mask & (df['SRCEARN'] == 3), 'INCFARM'] = df['INCLONGJ'] + df['OINCFARM']
    return df

def adjust_for_2019_topcode_jump(df):
    """Adjust for the topcode jump in 2019 for capital income."""
    topcode_limits = {
        'INCINT': 99999.,
        'INCRENT': 99999.,
        'INCDIVID': 100000.
    }
    for col, limit in topcode_limits.items():
        df.loc[df[col] > limit, col] = limit
    return df


def apply_topcoding_adjustment(df):
    # comment below from the stata code (although it doesnt look like its actually doing this)
    # this function just seems to be applying an adjustment factor, although wouldnt we want it to do that on all data, not just topcoded stuff?

    # *******************************************************
    # ********Create final topcoding for earnings************
    # **Don't touch post-1993 and just use real data*********
    # **For prior years use a Pareto imputation for topcode**
    # **Adj factor for 1993 based on 1994 data above 299999**
    # **Estimated to be 1.61 times AF for 1976 based on PK***
    # **tax data and E(Y|Y>=TC & Y<=2*TC) computed for*******
    # **1976 (TC of 50k) and 1994 ("artificial" TC of 100k)**
    # **Yields AF=1.85 in 1994 and AF=1.53 in 1976***********
    # **Further assume AF moves linearly from 1976 to 1993***
    # *******************************************************

    """Apply final topcoding for earnings directly in the original columns."""
    df['af'] = 1.53 + (1.85 - 1.53) * (df['YEAR'] - 1976) / (1994 - 1976)

    # 1976-1985: topcode at 99999
    for var in ['INCWAGE', 'INCBUS', 'INCFARM']:
        mask = (df['YEAR'] <= 1985) & (df[var] >= 99997) & (df[var] <= 100000)
        df.loc[mask, var] *= df['af']

    # 1986-1987: internal topcode at 250000
    for var in ['INCWAGE', 'INCBUS', 'INCFARM']:
        mask = df['YEAR'].between(1986, 1987) & (df[var] == 250000)
        df.loc[mask, var] *= df['af']


    # 1988-1993: internal topcode at 299999 for longest job
    mask = (df['YEAR'].between(1988, 1993)) & (df['INCLONGJ'].fillna(0) >= 299999) & (df['INCLONGJ'] <= 300000)
    df.loc[mask, 'INCLONGJ'] *= df['af']

    mask = df['YEAR'].between(1988, 1993)
    df.loc[mask & (df['SRCEARN'] == 1), 'INCWAGE'] = df['INCLONGJ'] + df['OINCWAGE']
    df.loc[mask & (df['SRCEARN'] == 2), 'INCBUS'] = df['INCLONGJ'] + df['OINCBUS']
    df.loc[mask & (df['SRCEARN'] == 3), 'INCFARM'] = df['INCLONGJ'] + df['OINCFARM']
    return df


def apply_topcoding_swaps(cps_df, swap_df):
    """Apply topcoding swaps and adjustments to the CPS data."""
    cps_df = merge_cps_with_swap_data(cps_df, swap_df)
    cps_df = fill_in_topcodes(cps_df)
    cps_df['INCLONGJ_2_1_FILL_IN_TOPCODES'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_2_1_FILL_IN_TOPCODES'] = cps_df['INCWAGE']
    cps_df = manually_fix_mistakes(cps_df)
    cps_df = reconstruct_wage_and_se_earnings_after_1988(cps_df)
    cps_df['INCLONGJ_2_2_RECONSTRUCT_AFTER_1988'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_2_2_RECONSTRUCT_AFTER_1988'] = cps_df['INCWAGE']
    cps_df = adjust_for_2019_topcode_jump(cps_df)
    cps_df = apply_topcoding_adjustment(cps_df)
    return cps_df

def adjust_incomes_for_inflation(df):
    """Adjust incomes to 2019 dollars using CPI99."""
    conversion_1999_to_2019 = 1.535
    for var in INCOME_VARS:
        df[var] = df[var] * df['CPI99'] * conversion_1999_to_2019
    return df

def adjust_capital_income_for_underreporting(df):
    """Adjust capital income for underreporting, from Rothbaum 2015a values."""
    df['INCINT'] = df['INCINT'] / 0.675
    df.loc[df['YEAR'] <= 1987, 'INCDRT'] = df['INCDRT'] / 0.414
    df.loc[df['YEAR'] > 1987, 'INCDIVID'] = df['INCDIVID'] / 0.695
    df.loc[df['YEAR'] > 1987, 'INCRENT'] = df['INCRENT'] / 0.274
    df.loc[df['YEAR'] > 1987, 'INCDRT'] = df['INCDIVID'] + df['INCRENT']
    return df

def filter_by_age(df):
    """Filter dataset by age range."""
    return df[(df['AGE'] >= HARD_MIN_AGE)]
# to replicate exactly: return df[(df['AGE'] >= HARD_MIN_AGE) & (df['AGE'] <= 65)]

def set_negative_rents_and_self_employed_income_to_0(df):
    """Set negative rents and self-employed income to zero."""
    for col in VARS_TO_SET_NONNEG:
        df[f'{col}_inclneg'] = df[col]
        df[f'{col}'] = df[col].clip(lower=0.)
    return df


def set_zero_hours_for_nonworkers(df):
    """Set usual hours to zero for non-workers."""
    df['UHRSWORKLY'] = np.where(df['UHRSWORKLY'] == 999, 0, df['UHRSWORKLY'])
    return df


def add_calculated_columns_for_income(df):
    df['capital_income'] = df[CAPITAL_INCOME_VARS].sum(axis=1)
    df['labor_income'] = df[LABOR_INCOME_VARS].sum(axis=1)
    df['total_income'] = df['capital_income'] + df['labor_income']
    df['income_year'] = df['YEAR'] - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df['log_capital_income'] = np.log(df['capital_income'].replace(0, np.nan))
        df['log_labor_income'] = np.log(df['labor_income'].replace(0, np.nan))
        df['log_total_income'] = np.log(df['total_income'].replace(0, np.nan))
        df['log_capital_over_labor'] = np.where(df['labor_income'] > 0, np.log(1 + df['capital_income'] / df['labor_income']), np.nan)
    return df


def flag_full_time_full_year(df):
    """Flag dataset for full-time, full-year workers."""
    df['yearly_hours_worked'] = (df['WKSWORK1'] * df['UHRSWORKLY']).astype(float)
    df['hourly_wage'] = np.divide(df['labor_income'], df['yearly_hours_worked'], out=np.zeros(df['yearly_hours_worked'].shape, dtype=float), where=df['labor_income']>0.)
    df['is_full_time_full_year'] = (df['WKSWORK1'] >= 49) & (df['UHRSWORKLY'] >= 40) & (df['hourly_wage'] >= 4)
    return df


def add_state_x_industry(df):
    df['stateXIND1990_singledigit'] = df['STATEFIP'].astype(str) + '-' + (df['IND1990'] // 100).astype(str)
    df['stateXIND1990'] = df['STATEFIP'].astype(str) + '-' + df['IND1990'].astype(str)
    df['stateXindustry_1950'] = df['STATEFIP'].astype(str) + '-' + df['IND1950'].astype(str)
    df['stateXindustry'] = df['STATEFIP'].astype(str) + '-' + df['IND'].astype(str)
    df['regionXIND1990'] = df['REGION'].astype(str) + '-' + df['IND1990'].astype(str)
    df['regionXindustry'] = df['REGION'].astype(str)[0] + '-' + df['IND'].astype(str)   #leftmost digit is region
    return df


def process_data(cps_df, swap_df):
    """Process a single chunk of CPS data."""
    cps_df = change_income_dtypes_to_float(cps_df)

    # Initial state logging for both INCLONGJ and INCWAGE
    cps_df['INCLONGJ_1_RAW'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_1_RAW'] = cps_df['INCWAGE']

    cps_df = fix_capital_income_topcoding(cps_df)
    cps_df = apply_topcoding_swaps(cps_df, swap_df)

    # Log after applying topcoding swaps
    cps_df['INCLONGJ_2_APPLY_TOPCODE_SWAPS'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_2_APPLY_TOPCODE_SWAPS'] = cps_df['INCWAGE']

    cps_df = filter_by_age(cps_df)

    # Log after filtering by age
    cps_df['INCLONGJ_3_FILTER_BY_AGE'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_3_FILTER_BY_AGE'] = cps_df['INCWAGE']

    cps_df = set_zero_hours_for_nonworkers(cps_df)

    # Log after setting zero hours for non-workers
    cps_df['INCLONGJ_4_ZERO_HOURS_NONWORKERS'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_4_ZERO_HOURS_NONWORKERS'] = cps_df['INCWAGE']

    cps_df = adjust_incomes_for_inflation(cps_df)

    # Log after adjusting for inflation
    cps_df['INCLONGJ_5_ADJUST_FOR_INFLATION'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_5_ADJUST_FOR_INFLATION'] = cps_df['INCWAGE']

    cps_df = set_negative_rents_and_self_employed_income_to_0(cps_df)

    # Log after setting negative rents/self-employed income to 0
    cps_df['INCLONGJ_6_NEG_RENTS_SELF_EMPLOYED_INCOME_0'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_6_NEG_RENTS_SELF_EMPLOYED_INCOME_0'] = cps_df['INCWAGE']

    cps_df = adjust_capital_income_for_underreporting(cps_df)

    # Log after adjusting capital income underreporting
    cps_df['INCLONGJ_7_ADJUST_CAP_UNDERREPORTING'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_7_ADJUST_CAP_UNDERREPORTING'] = cps_df['INCWAGE']

    cps_df = add_calculated_columns_for_income(cps_df)

    # Log after adding calculated columns for income
    cps_df['INCLONGJ_8_CALCULATED_COLUMNS_INCOME'] = cps_df['INCLONGJ']
    cps_df['INCWAGE_8_CALCULATED_COLUMNS_INCOME'] = cps_df['INCWAGE']

    cps_df = flag_full_time_full_year(cps_df)
    cps_df = add_state_x_industry(cps_df)

    return cps_df


def apply_filters(df, min_age=16, max_age=999, full_time_full_year_only=False, remove_top_1_percent=False):
    """Apply specified filters to the dataset."""
    if full_time_full_year_only:
        df = df[df['is_full_time_full_year']]
    if remove_top_1_percent:
        df = df[df['winsor99']]
    df = df[(df['AGE'] >= min_age) & (df['AGE'] <= max_age)]
    return df


def flag_top_1_percent(df):
    """Flag top 1% of income earners of full time full year workers."""
    for col in ['labor_income', 'capital_income']:
        for year in df['income_year'].unique():
            for sex in df['SEX'].unique():
                subset = df[(df['income_year'] == year) & (df['SEX'] == sex) & df['is_full_time_full_year']]
                top_1_percent_threshold = subset[col].quantile(0.99)
                df.loc[subset.index, f'{col}_99'] = top_1_percent_threshold

                df['labor_income_99'] = df.groupby(['income_year', 'SEX'])['labor_income'].transform(
                    lambda x: x.quantile(0.99))

    # somewhat counterintuitive that capital income is by labor income threshold but this is how the paper does it
    df['winsor99'] = (df['labor_income'] <= df['labor_income_99']) & (df['capital_income'] <= df['labor_income_99'])
    return df


# def read_data_chunked():
#     print("Processing swap data...")
#     swap_df = pd.read_csv(swap_filename)
#     swap_df = process_swap_data(swap_df)
#     print("Done!")
#
#     print("Setting up IPUMS Readers...")
#     ddi_codebook = readers.read_ipums_ddi(cps_codebook_filepath)
#     iter_microdata = readers.read_microdata_chunked(ddi_codebook, cps_filepath, chunksize=10000)
#     print("Done!")
#
#     year_data_dict = {}
#     chunk_num = 0
#
#     for chunk in iter_microdata:
#         print(f"\rProcessing year: {chunk['YEAR'].max()}", end="")
#         chunk_num += 1
#         chunk = chunk[chunk['YEAR'] >= MIN_DATA_YEAR]
#         if not chunk.empty:
#             processed_data = process_data(chunk, swap_df)
#             for year, group in processed_data.groupby('income_year'):
#                 if year in year_data_dict:
#                     year_data_dict[year] = pd.concat([year_data_dict[year], group])
#                 else:
#                     year_data_dict[year] = group
#
#     print("\nFinished processing CPS data")
#
#     # save_ data
#     print("Saving CPS data...")
#     for year, df in year_data_dict.items():
#         df = flag_top_1_percent(df)
#         filename = f'./data/cps_data/{str(year)}.csv'
#         df.to_csv(filename, index=False)
#
#
#     print("Done!")
def save_data(all_data):
    filename = f'../data/cps_data/all_years.csv'
    all_data.to_csv(filename, index=False)
    for year in range(MIN_DATA_YEAR, 2024):
        filename = f'./data/cps_data/{year}_sample.csv'
        year_data = all_data[all_data['YEAR'] == year]
        year_data.to_csv(filename, index=False)


def read_data():
    print("Processing swap data...")
    swap_df = pd.read_csv(swap_filename)
    swap_df = process_swap_data(swap_df)
    print("Done!")

    print("Setting up IPUMS Readers...")
    ddi_codebook = readers.read_ipums_ddi(cps_codebook_filepath)
    iter_microdata = readers.read_microdata_chunked(ddi_codebook, cps_filepath, chunksize=10000)
    print("Done!")

    all_data = pd.DataFrame()
    chunk_num = 0

    for chunk in iter_microdata:
        print(f"\rProcessing year: {chunk['YEAR'].max()}", end="")
        chunk_num += 1
        chunk = chunk[chunk['YEAR'] >= MIN_DATA_YEAR]
        if not chunk.empty:
            processed_data = process_data(chunk, swap_df)
            all_data = pd.concat([all_data, processed_data], ignore_index=True)

    all_data = flag_top_1_percent(all_data)
    print("\nFinished processing CPS data")

    print("Saving CPS data...")
    save_data(all_data)
    # save_ data
    print("Complete.")


if __name__ == "__main__":
    read_data()





