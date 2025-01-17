import numpy as np
from tqdm import tqdm
from .base_calculator import BaseCalculator


class GiniCalculator(BaseCalculator):
    def get_result_columns(self):
        """Define the columns for Gini coefficient results."""
        return ['Year', 'State', 'STATEFIP', 'exploitation', 'patronage',
                'exclusion', 'rationing', 'component_sum', 'total_gini']


    def calculate_n_and_mean_labor_income(self, df, weights=None):
        """Calculate the effective sample size and weighted mean labor income."""
        if weights is not None:
            # Weighted population size (sum of weights)
            total_weight = weights.sum()
            mean_labor_income = np.average(df['labor_income'], weights=weights)
        else:
            # Unweighted case: simple count and mean
            total_weight = len(df)
            mean_labor_income = df['labor_income'].mean()

        return total_weight, mean_labor_income

    def calculate_gini_denominator(self, df, weights=None):
        """
        Calculate the denominator for the Gini coefficient:
        2 * n^2 * mean_income
        """
        total_weight, mean_labor_income = self.calculate_n_and_mean_labor_income(df, weights)

        return 2 * total_weight ** 2 * mean_labor_income

    def calculate_exploitation_and_patronage_sums(self, df, group_identifier, weights=None, outer_tqdm=None):
        exploitation_sum = 0.
        patronage_sum = 0.
        all_pairs_sum = 0.

        grouped = df.groupby(group_identifier)
        total_groups = len(grouped)

        with tqdm(total=total_groups, desc="Calculating exploitation and patronage", leave=False,
                  position=outer_tqdm.position + 1 if outer_tqdm else 0) as pbar:
            for _, group in grouped:
                labor_income = group['labor_income'].values
                capital_income = group['capital_income'].values

                group_weights = weights.loc[group.index].values if weights is not None else np.ones(len(labor_income))

                labor_diff_matrix = labor_income[:, np.newaxis] - labor_income
                capital_diff_matrix = capital_income[:, np.newaxis] - capital_income

                domination_mask = capital_diff_matrix != 0
                higher_capital_higher_labor = (labor_diff_matrix * capital_diff_matrix > 0)

                weight_matrix = np.outer(group_weights, group_weights)

                exploitation_mask = domination_mask & higher_capital_higher_labor
                patronage_mask = domination_mask & ~higher_capital_higher_labor

                exploitation_sum += np.sum(
                    np.abs(labor_diff_matrix[exploitation_mask]) * weight_matrix[exploitation_mask]
                )
                patronage_sum += np.sum(
                    np.abs(labor_diff_matrix[patronage_mask]) * weight_matrix[patronage_mask]
                )

                all_pairs_sum += np.sum(
                    np.abs(labor_diff_matrix) * weight_matrix
                )

                pbar.update(1)

        return exploitation_sum, patronage_sum, all_pairs_sum

    def calculate_exclusion_and_rationing_sums(self, df, group_identifier, weights=None, outer_tqdm=None):
        exclusion_sum = 0.
        rationing_sum = 0.
        all_pairs_sum = 0.

        grouped = df.groupby(group_identifier)
        group_data = [
            (name, group['labor_income'].values, group['capital_income'].values,
             weights.loc[group.index].values if weights is not None else np.ones(len(group)))
            for name, group in grouped
        ]
        total_groups = len(group_data)

        with tqdm(total=total_groups * (total_groups - 1) // 2, desc="Calculating exclusion and rationing",
                  leave=False, position=outer_tqdm.position + 1 if outer_tqdm else 0) as pbar:
            for i in range(total_groups):
                _, group1_labor, group1_capital, group1_weights = group_data[i]
                for j in range(i + 1, total_groups):
                    _, group2_labor, group2_capital, group2_weights = group_data[j]

                    labor_diff = np.subtract.outer(group1_labor, group2_labor)
                    capital_diff = np.subtract.outer(group1_capital, group2_capital)

                    domination_mask = capital_diff != 0
                    higher_capital_higher_labor = (labor_diff * capital_diff > 0)

                    # Use survey weights
                    weight_matrix = np.outer(group1_weights, group2_weights)

                    exclusion_mask = domination_mask & higher_capital_higher_labor
                    rationing_mask = domination_mask & ~higher_capital_higher_labor

                    exclusion_sum += np.sum(
                        np.abs(labor_diff[exclusion_mask]) * weight_matrix[exclusion_mask]
                    )
                    rationing_sum += np.sum(
                        np.abs(labor_diff[rationing_mask]) * weight_matrix[rationing_mask]
                    )

                    all_pairs_sum += np.sum(
                        np.abs(labor_diff) * weight_matrix
                    )

                    pbar.update(1)

        # results * 2 for this function because we are only comparing each pair once for efficiency
        return exclusion_sum * 2, rationing_sum * 2, all_pairs_sum * 2

    def calculate_ginis(self, df, group_identifier, use_weights=False, outer_tqdm=None):
        """Calculate Gini coefficients with proper weight handling."""
        # Get weights if requested
        weights = df['ASECWT'] if use_weights else None

        # Calculate the denominator: 2 * n^2 * mean_income
        denominator = self.calculate_gini_denominator(df, weights)

        # Calculate the numerator sums for exploitation, patronage, exclusion, and rationing
        exploitation_sum, patronage_sum, pairs_sum_1 = self.calculate_exploitation_and_patronage_sums(
            df, group_identifier, weights, outer_tqdm)
        exclusion_sum, rationing_sum, pairs_sum_2 = self.calculate_exclusion_and_rationing_sums(
            df, group_identifier, weights, outer_tqdm)
        total_sum = pairs_sum_1 + pairs_sum_2

        if denominator == 0:
            raise ValueError("Denominator for Gini calculation is zero. Check the input data.")

        # Return the Gini coefficients
        return {
            'exploitation': exploitation_sum / denominator,
            'patronage': patronage_sum / denominator,
            'exclusion': exclusion_sum / denominator,
            'rationing': rationing_sum / denominator,
            'component_sum': (exploitation_sum + patronage_sum + exclusion_sum + rationing_sum) / denominator,
            'total_gini': total_sum / denominator   # distinct from component sum because it includes pairwise relationships with no pairwise domination
        }

    def calculate_for(self, df, **kwargs):
        """Calculate results for a specific year with proper weight handling."""
        group_identifier = kwargs.get('group_identifier', 'stateXindustry')
        use_weights = kwargs.get('use_weights', True)
        outer_tqdm = kwargs.get('outer_tqdm', None)

        gini_dict = self.calculate_ginis(df, group_identifier, use_weights, outer_tqdm)
        return gini_dict




