import pandas as pd
import numpy as np
from .base_calculator import BaseCalculator


class IncomeStatisticsCalculator(BaseCalculator):
    def calculate_statistics(self, df):
        weights = df['ASECWT']

        # Weighted calculations
        weighted_mean_log_labor = np.average(df['log_labor_income'], weights=weights)
        weighted_mean_log_total = np.average(df['log_total_income'], weights=weights)

        weighted_sq_diff_labor = np.average((df['log_labor_income'] - weighted_mean_log_labor) ** 2, weights=weights)
        weighted_sq_diff_total = np.average((df['log_total_income'] - weighted_mean_log_total) ** 2, weights=weights)

        n_weighted = len(df[df['ASECWT'] > 0])  # Count of observations with non-zero weights
        weighted_var_log_labor = weighted_sq_diff_labor * n_weighted / (n_weighted - 1) if n_weighted > 1 else np.nan
        weighted_var_log_total = weighted_sq_diff_total * n_weighted / (n_weighted - 1) if n_weighted > 1 else np.nan

        weighted_std_log_labor = np.sqrt(weighted_var_log_labor) if weighted_var_log_labor >= 0 else np.nan
        weighted_std_log_total = np.sqrt(weighted_var_log_total) if weighted_var_log_total >= 0 else np.nan

        # Unweighted calculations (explicitly labeled)
        unweighted_mean_log_labor = df['log_labor_income'].mean()
        unweighted_mean_log_total = df['log_total_income'].mean()

        unweighted_var_log_labor = df['log_labor_income'].var(ddof=1)  # Sample variance
        unweighted_var_log_total = df['log_total_income'].var(ddof=1)

        unweighted_std_log_labor = df['log_labor_income'].std(ddof=1)  # Sample standard deviation
        unweighted_std_log_total = df['log_total_income'].std(ddof=1)

        # Compile results
        stats = {
            # Weighted stats (default)
            'std_log_labor': weighted_std_log_labor,
            'std_log_total': weighted_std_log_total,
            'var_log_labor': weighted_var_log_labor,
            'var_log_total': weighted_var_log_total,
            'mean_log_labor': weighted_mean_log_labor,
            'mean_log_total': weighted_mean_log_total,

            # Unweighted stats (explicitly labeled)
            'unweighted_std_log_labor': unweighted_std_log_labor,
            'unweighted_std_log_total': unweighted_std_log_total,
            'unweighted_var_log_labor': unweighted_var_log_labor,
            'unweighted_var_log_total': unweighted_var_log_total,
            'unweighted_mean_log_labor': unweighted_mean_log_labor,
            'unweighted_mean_log_total': unweighted_mean_log_total,

            # Additional statistics
            'min_log_labor': np.min(df['log_labor_income']),
            'median_log_labor': np.median(df['log_labor_income']),
            'max_log_labor': np.max(df['log_labor_income']),
            'min_log_total': np.min(df['log_total_income']),
            'median_log_total': np.median(df['log_total_income']),
            'max_log_total': np.max(df['log_total_income']),
        }

        return stats

    def calculate_gender_statistics(self, df):
        stats = {}
        for sex, label in [(1, 'men'), (2, 'women')]:
            gender_df = df[df['SEX'] == sex]
            weights = gender_df['ASECWT']

            # Weighted calculations
            weighted_mean_log_labor = np.average(gender_df['log_labor_income'], weights=weights)
            weighted_mean_log_total = np.average(gender_df['log_total_income'], weights=weights)

            weighted_sq_diff_labor = np.average(
                (gender_df['log_labor_income'] - weighted_mean_log_labor) ** 2, weights=weights
            )
            weighted_sq_diff_total = np.average(
                (gender_df['log_total_income'] - weighted_mean_log_total) ** 2, weights=weights
            )

            n_weighted = len(gender_df[gender_df['ASECWT'] > 0])  # Count of non-zero weights
            weighted_var_log_labor = (
                weighted_sq_diff_labor * n_weighted / (n_weighted - 1) if n_weighted > 1 else np.nan
            )
            weighted_var_log_total = (
                weighted_sq_diff_total * n_weighted / (n_weighted - 1) if n_weighted > 1 else np.nan
            )

            weighted_std_log_labor = np.sqrt(weighted_var_log_labor) if weighted_var_log_labor >= 0 else np.nan
            weighted_std_log_total = np.sqrt(weighted_var_log_total) if weighted_var_log_total >= 0 else np.nan

            # Unweighted calculations (explicitly labeled)
            unweighted_mean_log_labor = gender_df['log_labor_income'].mean()
            unweighted_mean_log_total = gender_df['log_total_income'].mean()

            unweighted_var_log_labor = gender_df['log_labor_income'].var(ddof=1)
            unweighted_var_log_total = gender_df['log_total_income'].var(ddof=1)

            unweighted_std_log_labor = gender_df['log_labor_income'].std(ddof=1)
            unweighted_std_log_total = gender_df['log_total_income'].std(ddof=1)

            # Compile results
            stats.update({
                # Weighted stats
                f'{label}_std_log_labor': weighted_std_log_labor,
                f'{label}_std_log_total': weighted_std_log_total,
                f'{label}_var_log_labor': weighted_var_log_labor,
                f'{label}_var_log_total': weighted_var_log_total,
                f'{label}_mean_log_labor': weighted_mean_log_labor,
                f'{label}_mean_log_total': weighted_mean_log_total,

                # Unweighted stats
                f'unweighted_{label}_std_log_labor': unweighted_std_log_labor,
                f'unweighted_{label}_std_log_total': unweighted_std_log_total,
                f'unweighted_{label}_var_log_labor': unweighted_var_log_labor,
                f'unweighted_{label}_var_log_total': unweighted_var_log_total,
                f'unweighted_{label}_mean_log_labor': unweighted_mean_log_labor,
                f'unweighted_{label}_mean_log_total': unweighted_mean_log_total,

                # Additional statistics
                f'{label}_min_log_labor': np.min(gender_df['log_labor_income']),
                f'{label}_max_log_labor': np.max(gender_df['log_labor_income']),
                f'{label}_min_log_total': np.min(gender_df['log_total_income']),
                f'{label}_max_log_total': np.max(gender_df['log_total_income']),
                f'{label}_median_log_labor': np.median(gender_df['log_labor_income']),
                f'{label}_median_log_total': np.median(gender_df['log_total_income']),
            })

        return stats

    def get_result_columns(self):
        """Define the columns for income statistics results."""
        base_metrics = ['std_log_labor', 'std_log_total', 'var_log_labor', 'var_log_total',
                        'mean_log_labor', 'mean_log_total', 'min_log_labor', 'median_log_labor',
                        'max_log_labor', 'min_log_total', 'median_log_total', 'max_log_total']

        # Add unweighted versions
        unweighted = [f'unweighted_{metric}' for metric in base_metrics]

        # Add gender-specific versions
        gender_metrics = []
        for gender in ['men', 'women']:
            gender_metrics.extend([f'{gender}_{metric}' for metric in base_metrics])
            gender_metrics.extend([f'unweighted_{gender}_{metric}' for metric in base_metrics])

        # Combine all columns
        return ['Year', 'State', 'STATEFIP'] + base_metrics + unweighted + gender_metrics


    def calculate_for(self, df, **kwargs):
        stats = self.calculate_statistics(df)
        stats.update(self.calculate_gender_statistics(df))
        return stats

