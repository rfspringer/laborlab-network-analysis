import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
import network_sort
import utils
from scipy.stats import spearmanr
import pandas as pd


class Simulation:
    def __init__(
        self,
        num_nodes: int,
        prob_connection: float = 0.5,
        wealth_dist=None,
        wealth_to_income_fn=lambda x: x * 0.2,
        **wealth_kwargs
    ):
        """
        Initializes a new graph simulation
        :param num_nodes: Number of nodes for simulated graph
        :param prob_connection: Probability of each pair of nodes being connected in simulation
        :param wealth_dist: Distribution function for wealth, ex. np.random.normal, np.random.randint. By default uses randint from 0 - 1000000
        :param wealth_to_income_fn: Function for calculating income from wealth. By default 0.2 * wealth
        :param wealth_kwargs: Args for provided wealth distribution function
        """
        self.id_to_wealth = self.make_id_to_wealth_dict(
            num_nodes, wealth_dist, **wealth_kwargs
        )
        self.id_to_income = self.make_id_to_income_dict(wealth_to_income_fn)
        self.true_sorted_ids, self.ranks_by_id = self.sort_ids_by_wealth()
        self.G = self.make_graph(num_nodes, prob_connection)
        self.sorted_nodes = network_sort.sort(
            self.G
        )  # note this comes out at list of sets (in case of ties), not list of elements
        self.rank_correlation, self.best_rank_correlation_list = (
            self.calculate_rank_correlation()
        )

    def make_id_to_wealth_dict(self, num_nodes, wealth_dist, **kwargs) -> {int: float}:
        """
        Make dictionary of node ids to wealth, generated from wealth_dist function
        :param num_nodes: Number of nodes in the simulation
        :param wealth_dist: Distribution to use for generating wealth. By default np.random.randint from 0 - 1000000
        :param kwargs: Args for wealth_dist function.
        ex {"low": ____, "high": ____} for randint, {"loc": ___, "scale": ___} for normal
        :return: Dictionary of node ids to wealth
        """

        # Set default distribution function to np.random.randint with low=0 and high=100000
        if wealth_dist is None:
            wealth_dist = np.random.randint
            kwargs.setdefault("low", 0)
            kwargs.setdefault("high", 1000000)

        wealths = wealth_dist(size=num_nodes, **kwargs)
        return {i: wealths[i] for i in range(num_nodes)}

    def make_id_to_income_dict(self, wealth_to_income_fn) -> {int: float}:
        """
        Create id to income dictionary based from wealth-to-income function
        :param wealth_to_income_fn: function for converting wealth to income
        :return: Dictionary of node ids to income
        """
        return {
            id: wealth_to_income_fn(wealth) for id, wealth in self.id_to_wealth.items()
        }

    def sort_ids_by_wealth(self) -> ([int], {int: int}):
        """
        Sort simulated nodes by wealth
        :return: list of ids sorted by wealth, dictionary of node ids to rank
        """
        sorted_ids = sorted(
            self.id_to_wealth.keys(), key=lambda k: self.id_to_wealth[k], reverse=True
        )
        ranks = {id_: rank for rank, id_ in enumerate(sorted_ids, start=1)}
        return sorted_ids, ranks

    def make_graph(self, num_nodes: int, prob_connection: float) -> nx.DiGraph:
        """
        Build graph from graph parameters and incomes
        :param num_nodes: Number of nodes for graph to have
        :param prob_connection: Probability of an edge between any two nodes
        :return: Simulated weighted directed graph of nodes, with weights/ direction of income difference
        """
        # create list of all pairs of ids
        pairs = list(itertools.combinations(self.id_to_income.keys(), 2))

        # make an empty graph
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))

        for pair in pairs:
            # choose whether or not nodes should be connected with probability prob_connection
            connected = np.random.choice(
                [0, 1], p=[1 - prob_connection, prob_connection]
            )
            if connected:
                self.add_edge(G, pair)
        return G

    def add_edge(self, G: nx.DiGraph, node_pair: (int, int)):
        """
        Add an edge between two nodes, pointing from higher income node to lower income node, with weight of difference in incomes
        :param G: Graph to add edge to
        :param node_pair: tuple of two node ids to connect
        :return:
        """
        a, b = node_pair[0], node_pair[1]
        weight = self.id_to_income[a] - self.id_to_income[b]

        # add edge with weight of income difference, pointing in direction from higher to lower income
        # if incomes equal, direction is effectively random
        if weight > 0:
            G.add_edge(a, b, weight=weight)
        else:
            G.add_edge(b, a, weight=-weight)

    def show(self):
        """
        Display a visualization of the simulated graph
        :return:
        """
        positions = nx.circular_layout(self.G)  # Compute node positions

        # Draw nodes and edges
        nx.draw(
            self.G,
            pos=positions,
            with_labels=True,
            node_color="skyblue",
            node_size=1500,
            font_size=12,
            font_color="black",
        )
        edge_labels = {(u, v): self.G[u][v]["weight"] for u, v in self.G.edges()}
        nx.draw_networkx_edge_labels(
            self.G, pos=positions, edge_labels=edge_labels, font_color="black"
        )

        plt.show()

    def calculate_rank_correlation(self) -> (float, [int]):
        """
        Calculate spearman rank correlation between true sorted order of wealth and results from sorting algorithm
        :return: spearman rank correlation between true order and sorting algorithm results, list used to calculate result
        """

        # all possible orderings of elements from the sorting algorithm (in case of ties, though this is infrequent unless weights are often the same)
        all_lists = utils.get_list_of_set_permutations(self.sorted_nodes)

        # checks through all possible orders within sets to get closest to correct one (since all would be valid orderings)
        best_list = None
        max_rank_correl = -1
        for list in all_lists:
            list_ranks = [self.ranks_by_id[node] for node in list]
            true_ranks = [self.ranks_by_id[node] for node in self.true_sorted_ids]
            rho, p_value = spearmanr(list_ranks, true_ranks)
            if rho > max_rank_correl:
                max_rank_correl = rho
                best_list = list
        return max_rank_correl, best_list


def run_simulations_partially_connected_no_noise():
    num_nodes_values = [10, 50, 100, 500, 1000]
    prob_connected_values = [0.2, 0.35, 0.5, 0.75]

    param_combinations = [
        (p1, p2) for p1 in num_nodes_values for p2 in prob_connected_values
    ]

    # Run the function with each set of parameters and collect results
    results = []
    i = 1
    sims_per_combination = 10
    for param1, param2 in param_combinations:
        j = 0
        for j in range(sims_per_combination):
            update_str = (
                str(i)
                + "/"
                + str(
                    len(num_nodes_values)
                    * len(prob_connected_values)
                    * sims_per_combination
                )
            )
            print(update_str)
            result = Simulation(param1, param2).rank_correlation
            results.append((param1, param2, result))
            i += 1

    # Create a DataFrame from the results
    df_results = pd.DataFrame(results, columns=["num_nodes", "p", "rank_correl"])
    df_results.to_csv(
        "simulation_results_partially_connected_no_noise.csv", index=False
    )


def run_simulations_all_edges_with_noise():
    num_nodes_values = [10, 50, 100, 500, 1000]

    # Run the function with each set of parameters and collect results
    results = []
    i = 1
    sims_per_combination = 10
    for n in num_nodes_values:
        for j in range(sims_per_combination):
            update_str = (
                str(i) + "/" + str(len(num_nodes_values) * sims_per_combination)
            )
            print(update_str)

            simulation = Simulation(
                n,
                prob_connection=1,
                wealth_to_income_fn=lambda x: 0.2 * x + np.random.normal(scale=10000),
            )
            result = simulation.rank_correlation
            results.append((n, result))
            i += 1

    # Create a DataFrame from the results
    df_results = pd.DataFrame(results, columns=["num_nodes", "rank_correl"])
    df_results.to_csv("simulation_results_fully_connected_noise.csv", index=False)


run_simulations_all_edges_with_noise()
run_simulations_partially_connected_no_noise()

# Code for running one simualtion, checking outputs
# sim = Simulation(num_nodes=10, wealth_to_income_fn= lambda x: 0.2 * x + np.random.normal(scale=10000))
# print("Id to wealth: ", sim.id_to_wealth)
# print("Id to income: ", sim.id_to_income)
# print("True order: ", sim.true_sorted_ids)
# print("Sorted order in sets: ", sim.sorted_nodes)
# print("Closest order: ", sim.best_rank_correlation_list)
# print("rank correl: ", sim.rank_correlation)
# sim.show()
