import networkx as nx
from itertools import combinations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Any


# Example usage:
# gini_calculator = GiniCalculatorFromNetX(income_graph=G_income, domination_graph=G_domination)
#
# # Calculate Gini components using the directed edge domination strategy
# results = gini_calculator.calculate_components(DominationStrategies.directed_edge)
#
# # Save the results to a CSV file
# gini_calculator.save_results(results, 'gini_results.csv')


class DominationStrategies:
    # container class to hold different ways to measure/ check for domination
    @staticmethod
    def directed_edge(g: nx.DiGraph, u: Any, v: Any, **kwargs) -> bool:
        """Check if there's a directed edge from u to v."""
        return g.has_edge(u, v)

    @staticmethod
    def attribute_comparison(g: nx.Graph, u: Any, v: Any, *, domination_attr: str, **kwargs) -> bool:
        """Check if node u dominates node v based on a higher attribute value."""
        return float(g.nodes[u].get(domination_attr)) > float(g.nodes[v].get(domination_attr))

    # could add others if useful/ interesting

    @staticmethod
    def custom_rule(g: nx.Graph, u: Any, v: Any, *, rule_func: Callable, **kwargs) -> bool:
        """Apply a custom domination rule provided as a function."""
        return rule_func(g, u, v, **kwargs)


class GiniCalculatorFromNetX:
    def __init__(self,
                 income_graph: nx.Graph,
                 income_attr: str = 'income',
                 domination_graph: Optional[nx.Graph] = None):
        """
        Initialize the Gini calculator with an income graph and optional separate domination graph.

        Args:
            income_graph: NetworkX graph with income attributes
            income_attr: Name of the income attribute in the graph
            domination_graph: Optional separate graph for checking domination relationships
        """
        self.g_income = income_graph
        self.g_domination = domination_graph if domination_graph is not None else income_graph
        self.income_attr = income_attr
        self.denominator = self._calculate_denominator()

    def _calculate_denominator(self) -> float:
        """Calculate the Gini coefficient denominator."""
        n = len(self.g_income.nodes())
        income_values = np.array([
            data.get(self.income_attr, 0)
            for _, data in self.g_income.nodes(data=True)
        ])
        mean_income = np.mean(income_values) if n > 0 else 0
        return 2 * (n ** 2) * mean_income

    def calculate_components(
            self,
            domination_fn: Callable,
            **domination_params
    ) -> Dict[str, float]:
        """
        Calculate Gini components using the provided domination function.

        Args:
            domination_check: Function that determines if one node dominates another
            **domination_params: Additional parameters to pass to the domination check

        Returns:
            Dictionary containing Gini components and total Gini
        """
        sums = {
            'exploitation_sum': 0.,
            'patronage_sum': 0.,
            'exclusion_sum': 0.,
            'rationing_sum': 0.,
            'all_pairs_sum': 0.
        }

        for u, v in combinations(self.g_income.nodes(), 2):
            # Skip if nodes don't exist in domination graph
            if not (self.g_domination.has_node(u) and self.g_domination.has_node(v)):
                continue

            income_u = float(self.g_income.nodes[u].get(self.income_attr, 0))
            income_v = float(self.g_income.nodes[v].get(self.income_attr, 0))
            income_diff = abs(income_u - income_v)

            # Use domination graph for domination checks
            u_dominates_v = domination_fn(self.g_domination, u, v, **domination_params)
            v_dominates_u = domination_fn(self.g_domination, v, u, **domination_params)

            # Use income graph for edge checks
            if (u_dominates_v and income_u > income_v) or (v_dominates_u and income_v > income_u):
                if self.g_income.has_edge(u, v):
                    sums['exploitation_sum'] += 2 * income_diff
                else:
                    sums['exclusion_sum'] += 2 * income_diff
            elif (u_dominates_v and income_u < income_v) or (v_dominates_u and income_v < income_u):
                if self.g_income.has_edge(u, v):
                    sums['patronage_sum'] += 2 * income_diff
                else:
                    sums['rationing_sum'] += 2 * income_diff

            sums['all_pairs_sum'] += 2 * income_diff

        results = {
            'exploitation': sums['exploitation_sum'] / self.denominator,
            'patronage': sums['patronage_sum'] / self.denominator,
            'exclusion': sums['exclusion_sum'] / self.denominator,
            'rationing': sums['rationing_sum'] / self.denominator,
            'component_sum': sum(
                sums[k] for k in ['exploitation_sum', 'patronage_sum', 'exclusion_sum', 'rationing_sum']
            ) / self.denominator,
            'total_gini': sums['all_pairs_sum'] / self.denominator
        }

        return results

    def save_results(self, results: Dict[str, float], filename: str) -> None:
        """Save results to a CSV file."""
        pd.DataFrame([results]).to_csv(filename, index=False)