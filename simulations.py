import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import network_sort
import utils
import pandas as pd
from graph_analysis import *


class Simulation:
    def __init__(
        self,
        num_nodes: int,
        degree_dist=np.random.randint,
        wealth_dist=lambda n : np.random.randint(low=1000, high=100000),
        wealth_to_income_fn=lambda x: x * 0.2,
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
            num_nodes, wealth_dist
        )
        self.id_to_income = self.make_id_to_income_dict(wealth_to_income_fn)
        self.true_sorted_ids, self.ranks_by_id = self.sort_ids_by_wealth()
        degrees = self.generate_degrees(degree_dist, num_nodes)
        self.G = self.make_graph(degrees)
        self.sorted_nodes = network_sort.sort(
            self.G
        )  # note this comes out at list of sets (in case of ties), not list of elements
        self.rank_correlation, self.best_rank_correlation_list = \
            utils.calculate_rank_correlation(self.sorted_nodes, self.ranks_by_id)

    def make_id_to_wealth_dict(self, num_nodes, wealth_dist) -> {int: float}:
        """
        Make dictionary of node ids to wealth, generated from wealth_dist function
        :param num_nodes: Number of nodes in the simulation
        :param wealth_dist: Distribution to use for generating wealth. By default np.random.randint from 0 - 1000000
        :param kwargs: Args for wealth_dist function.
        ex {"low": ____, "high": ____} for randint, {"loc": ___, "scale": ___} for normal
        :return: Dictionary of node ids to wealth
        """

        # Set default distribution function to np.random.randint with low=0 and high=100000
        wealths = wealth_dist(num_nodes)
        return {i: wealths[i] for i in range(num_nodes)}

    def make_id_to_income_dict(self, wealth_to_income_fn) -> {int: float}:
        """
        Create id to income dictionary based from wealth-to-income function
        :param wealth_to_income_fn: function for converting wealth to income
        :return: Dictionary of node ids to income
        """
        dict = {
            id: wealth_to_income_fn(wealth) for id, wealth in self.id_to_wealth.items()
        }
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

    def generate_degrees(self, degree_generator, num_nodes) -> [int]:
        degrees = degree_generator(num_nodes) # make sure the generator takes things in this format!!

        # Make the sum of degrees is even so it can make a valid graph
        if sum(degrees) % 2 != 0:
            degrees[np.random.choice(num_nodes)] += 1

        return degrees

    def make_graph(self, degrees: [int]) -> nx.DiGraph:
        """
        Build graph from graph parameters and incomes
        :param num_nodes: Number of nodes for graph to have
        :return: Simulated weighted directed graph of nodes, with weights/ direction of income difference
        """
        # make an undirected graph from the specified degrees
        G_undirected = nx.expected_degree_graph(degrees, selfloops=False)
        nx.set_node_attributes(G_undirected, self.id_to_wealth, 'wealth')
        nx.set_node_attributes(G_undirected, self.id_to_income, 'income')

        # convert to directed graph, edges pointing from higher to lower wealth node
        G = nx.DiGraph()
        G.add_nodes_from(G_undirected.nodes())

        for u, v in G_undirected.edges():
            self.add_edge(G, (u, v))
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


def run_simulations_partially_connected_no_noise():
    num_nodes_values = [10, 50, 100, 500, 1000]
    prob_connected_values = [0.2, 0.35, 0.5, 0.65, 0.8]

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
        "results/simulation_results_partially_connected_no_noise.csv", index=False
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
                wealth_to_income_fn=lambda x: 0.2 * x + np.random.normal(scale=50000),
            )
            rho = simulation.rank_correlation
            wealth_gini = wealth_gini_all_nodes(simulation.G)
            directly_connected_gini = wealth_gini_directly_connected_only(simulation.G)
            not_connected_gini = wealth_gini_not_directly_connected(simulation.G)
            income_wealth_match_gini, income_wealth_no_match_gini = wealth_gini_directly_connected_split_by_income(simulation.G)
            results.append((n, rho, wealth_gini, directly_connected_gini, not_connected_gini, income_wealth_match_gini, income_wealth_no_match_gini))
            i += 1

    # Create a DataFrame from the results
    df_results = pd.DataFrame(results, columns=["num_nodes", "rank correl", "wealth gini", "gini- directly connected",
                                                "gini-not connected", "gini- income-wealth match", "gini- no income wealth match"])
    df_results.to_csv("results/simulation_results_fully_connected_higher_noise.csv", index=False)



def run_simulations():
    num_nodes_values = [10, 50, 100, 500, 1000]
    prob_connected_values = [0.2, 0.35, 0.5, 0.65, 0.8]
    noise_scales = [0, 0.05, 0.1, 0.25, 0.5]

    param_combinations = [
        (p1, p2, p3 ) for p1 in num_nodes_values for p2 in prob_connected_values for p3 in noise_scales
    ]

    # Run the function with each set of parameters and collect results
    results = []
    i = 1
    sims_per_combination = 10
    for num_nodes, prob_connected, noise_scale in param_combinations:
        for j in range(sims_per_combination):
            update_str = (
                str(i)
                + "/"
                + str(
                    len(num_nodes_values)
                    * len(prob_connected_values)
                    * len(noise_scales)
                    * sims_per_combination
                )
            )
            print(update_str)
            simulation = Simulation(num_nodes=num_nodes, wealth_dist=None, wealth_to_income_fn=lambda x: x * 0.2 + np.random.normal(scale= 0.2 * x * noise_scale))
            rho = simulation.rank_correlation
            wealth_gini = wealth_gini_all_nodes(simulation.G)
            directly_connected_gini = wealth_gini_directly_connected_only(simulation.G)
            not_connected_gini = wealth_gini_not_directly_connected(simulation.G)
            weakly_connected_gini = wealth_gini_weakly_connected_only(simulation.G)
            not_weakly_connected_gini = wealth_gini_weakly_unconnected_only(simulation.G)
            income_wealth_match_gini, income_wealth_no_match_gini = wealth_gini_directly_connected_split_by_income(
                simulation.G)
            results.append((num_nodes, prob_connected, noise_scale, rho, wealth_gini, directly_connected_gini, not_connected_gini, weakly_connected_gini, not_weakly_connected_gini, income_wealth_match_gini,
                            income_wealth_no_match_gini))
            i += 1

    df_results = pd.DataFrame(results, columns=["num_nodes", "p", "noise_scale", "rank_correl", "wealth gini", "gini- directly connected",
                                                "gini- not directly connected", "gini- weakly connected",
                                                "gini- not weakly connected", "gini- income-wealth match", "gini- no income wealth match"])
    df_results.to_csv(
        "results/simulation_results_all.csv", index=False
    )

def run_simulations_varying_correlation():
    num_nodes_values = [100]
    mins = [np.array([2., 1000.])]
    alpha = [np.array([2., 2.])]
    correlations = [correlation_matrix = np.array([[1., 0.5], [0.5, 1.]])]

    noise_scales = [0, 0.05, 0.1, 0.25, 0.5]

    param_combinations = [
        (p1, p2, p3 ) for p1 in num_nodes_values for p2 in prob_connected_values for p3 in noise_scales
    ]

    # Run the function with each set of parameters and collect results
    results = []
    i = 1
    sims_per_combination = 10
    for num_nodes, prob_connected, noise_scale in param_combinations:
        for j in range(sims_per_combination):
            update_str = (
                str(i)
                + "/"
                + str(
                    len(num_nodes_values)
                    * len(prob_connected_values)
                    * len(noise_scales)
                    * sims_per_combination
                )
            )
            print(update_str)
            simulation = Simulation(num_nodes=num_nodes, wealth_dist=None, wealth_to_income_fn=lambda x: x * 0.2 + np.random.normal(scale= 0.2 * x * noise_scale))
            rho = simulation.rank_correlation
            wealth_gini = wealth_gini_all_nodes(simulation.G)
            directly_connected_gini = wealth_gini_directly_connected_only(simulation.G)
            not_connected_gini = wealth_gini_not_directly_connected(simulation.G)
            weakly_connected_gini = wealth_gini_weakly_connected_only(simulation.G)
            not_weakly_connected_gini = wealth_gini_weakly_unconnected_only(simulation.G)
            income_wealth_match_gini, income_wealth_no_match_gini = wealth_gini_directly_connected_split_by_income(
                simulation.G)
            results.append((num_nodes, prob_connected, noise_scale, rho, wealth_gini, directly_connected_gini, not_connected_gini, weakly_connected_gini, not_weakly_connected_gini, income_wealth_match_gini,
                            income_wealth_no_match_gini))
            i += 1

    df_results = pd.DataFrame(results, columns=["num_nodes", "p", "noise_scale", "rank_correl", "wealth gini", "gini- directly connected",
                                                "gini- not directly connected", "gini- weakly connected",
                                                "gini- not weakly connected", "gini- income-wealth match", "gini- no income wealth match"])
    df_results.to_csv(
        "results/simulation_results_all.csv", index=False
    )

# def run_pareto_simulations():
#     # want to to adjust alpha and correl matrix
#     # and maybe num nodes too?
#     # and noise?



#Code for running one simualtion, checking outputs

# wealth/ degree distributions:
degree_dist_uniform = lambda n, low=2, high=5: np.random.randint(low, high + 1, size=n)
wealth_dist_uniform = lambda n, low=1000, high=1000000: np.random.randint(low, high + 1, size=n)

mins = np.array([2., 1000.])
alpha = np.array([2., 2.])
correlation_matrix = np.array([[1., 0.5], [0.5, 1.]])
n_samples=100
pareto_samples = utils.multivariate_pareto_dist(n_samples, mins, alpha, correlation_matrix)
degree_dist = lambda n: np.round(pareto_samples[:n, 0])
wealth_dist = lambda n: pareto_samples[:n, 1]

sim = Simulation(num_nodes=10, degree_dist=degree_dist, wealth_dist=wealth_dist)
print("Id to wealth: ", sim.id_to_wealth)
print("Id to income: ", sim.id_to_income)
print("True order: ", sim.true_sorted_ids)
print("Sorted order in sets: ", sim.sorted_nodes)
print("Closest order: ", sim.best_rank_correlation_list)
print("rank correl: ", sim.rank_correlation)
sim.show()




