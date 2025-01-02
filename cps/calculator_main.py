import logging
from cps.cps_data_calculators import GiniCalculator


def main():
    logging.basicConfig(level=logging.INFO)

    # Calculate Gini coefficients
    gini_calc = GiniCalculator('../results/weighted_gini_components_by_ind_18plus_1976plus.json')
    gini_calc.calculate_all(
        min_age=18,
        # filter_full_time_full_year=True,
        filter_ag_and_public_service=True,
        min_year=1976,
        parallel=True,
        run_from_existing_file=True,
        use_weights=True,
        group_identifier='stateXIND1990'
    )

    # # Calculate income statistics
    # stats_calc = IncomeStatisticsCalculator('results/income_statistics_26to65_1976plus_ftfy.json')
    # stats_calc.calculate_all(
    #     min_age=26,
    #     max_age=65,
    #     min_year=1976,
    #     filter_full_time_full_year=True,
    #     filter_top_1_percent=False,
    #     filter_ag_and_public_service=False,
    #     parallel=True
    # )


if __name__ == "__main__":
    main()