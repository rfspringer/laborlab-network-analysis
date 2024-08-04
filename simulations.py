import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import network_sort
import utils
import pandas as pd
from graph_analysis_from_net import *
import matplotlib.cm as cm


class Simulation:
    def __init__(
        self,
        num_nodes: int,
        degree_dist=np.random.randint,
        wealth_dist=lambda n : np.random.randint(low=1000, high=100000),
        wealth_to_income_fn=lambda x: 0.2 * x * np.random.lognormal(mean=0, sigma=0.1),
    ):
        """
        Initializes a new graph simulation
        :param num_nodes: Number of nodes for simulated graph
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

        # convert to directed graph, edges pointing from higher to lower wealth node
        G = nx.DiGraph()
        G.add_nodes_from(G_undirected.nodes())
        nx.set_node_attributes(G, self.id_to_wealth, 'wealth')
        nx.set_node_attributes(G, self.id_to_income, 'income')

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
        else :
            G.add_edge(b, a, weight=-weight)

    def show(self):
        """
        Display a visualization of the simulated graph
        :return:
        """
        positions = nx.circular_layout(self.G)  # Compute node positions

        # Get wealth values and normalize them
        wealth_values = np.array([self.G.nodes[node].get('wealth', 0) for node in self.G.nodes()])
        wealth_min, wealth_max = wealth_values.min(), wealth_values.max()
        normalized_wealth = (wealth_values - wealth_min) / (wealth_max - wealth_min)

        # Create a colormap for nodes
        node_colors = plt.cm.Blues(normalized_wealth)

        # Get edge weights and normalize them
        edge_weights = np.array([self.G[u][v].get('weight', 1) for u, v in self.G.edges()])
        weight_min, weight_max = 0, edge_weights.max()
        normalized_weights = (edge_weights - weight_min) / (weight_max - weight_min)

        # Create a colormap for edges
        edge_colors = plt.cm.Greys(normalized_weights)

        # Create node labels with wealth and income
        node_labels = {
            node: f"Wealth: {int(self.G.nodes[node].get('wealth'))}\nIncome: {int(self.G.nodes[node].get('income'))}"
            for node in self.G.nodes()
        }

        # Draw nodes
        nx.draw_networkx_nodes(
            self.G,
            pos=positions,
            node_color=node_colors,
            node_size=4000,
            edgecolors = 'black',
        )

        # Draw edges with colors based on weights
        nx.draw_networkx_edges(
            self.G,
            pos=positions,
            edge_vmin=weight_min,
            edge_vmax=weight_max
        )

        # Draw labels for nodes
        nx.draw_networkx_labels(
            self.G,
            pos=positions,
            labels=node_labels,
            font_size=8,
            font_color="black"
        )

        # Draw edge labels
        edge_labels = {(u, v): int(self.G[u][v]["weight"]) for u, v in self.G.edges()}
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
            directly_connected_gini = wealth_gini_directly_connected(simulation.G)
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
            directly_connected_gini = wealth_gini_directly_connected(simulation.G)
            not_connected_gini = wealth_gini_not_directly_connected(simulation.G)
            weakly_connected_gini = wealth_gini_weakly_connected(simulation.G)
            not_weakly_connected_gini = wealth_gini_not_weakly_connected(simulation.G)
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


def run_simulations_varying_correlation(num_nodes_values, min_values, alpha_values, correlation_values, noise_scales, sims_per_combination):
    param_combinations = [
        (p1, p2, p3, p4, p5) for p1 in num_nodes_values for p2 in min_values for p3 in alpha_values for p4 in correlation_values for p5 in noise_scales
    ]

    # Run the function with each set of parameters and collect results
    results = []
    i = 1
    sims_per_combination = sims_per_combination
    total_combinations = len(num_nodes_values) * len(min_values) * len(alpha_values) * len(correlation_values) * len(noise_scales) * sims_per_combination
    for num_nodes, min_val, alpha, correlation, noise_scale in param_combinations:
        for j in range(sims_per_combination):
            # print update string
            print(f"\r{i}/{total_combinations}", end='')

            # get degree and wealth distributions from parameters
            pareto_samples = utils.multivariate_pareto_dist(num_nodes, min_val, alpha, correlation)
            degree_dist = lambda n: np.round(pareto_samples[:n, 0])
            wealth_dist = lambda n: pareto_samples[:n, 1]

            # set up simulation
            simulation = Simulation(
                num_nodes=num_nodes,
                degree_dist=degree_dist,
                wealth_dist=wealth_dist,
                wealth_to_income_fn= lambda x: 0.2 * x * np.random.lognormal(mean=0, sigma=noise_scale)
            )

            # save results
            rho = simulation.rank_correlation
            wealth_gini = wealth_gini_all_nodes(simulation.G)
            directly_connected_gini = wealth_gini_directly_connected(simulation.G)
            not_connected_gini = wealth_gini_not_directly_connected(simulation.G)
            weakly_connected_gini = wealth_gini_weakly_connected(simulation.G)
            not_weakly_connected_gini = wealth_gini_not_weakly_connected(simulation.G)
            income_wealth_match_gini, income_wealth_no_match_gini = wealth_gini_directly_connected_split_by_income(
                simulation.G)

            results.append((num_nodes, min_val[0], min_val[1], alpha[0], alpha[1], correlation, noise_scale, rho, wealth_gini, directly_connected_gini, not_connected_gini, weakly_connected_gini, not_weakly_connected_gini, income_wealth_match_gini,
                            income_wealth_no_match_gini))
            i += 1

    df_results = pd.DataFrame(results, columns=["num_nodes", "min_degree", "min_wealth", "alpha_degree", "alpha_wealth", "correlation", "noise_scale", "rank_correl", "wealth gini", "gini- directly connected",
                                                "gini- not directly connected", "gini- weakly connected",
                                                "gini- not weakly connected", "gini- income-wealth match", "gini- no income wealth match"])
    df_results.to_csv(
        "results/simulation_results_correlations_0-1.csv", index=False
    )


def run_one_sim():
    # Code for running one simualtion, checking outputs

    mins = np.array([2., 1000.])
    alpha = np.array([2., 2.])
    correlation = 0.
    n_samples = 1000
    pareto_samples = utils.multivariate_pareto_dist(n_samples, mins, alpha, correlation)
    degree_dist = lambda n: np.round(pareto_samples[:n, 0])
    wealth_dist = lambda n: pareto_samples[:n, 1]

    sim = Simulation(num_nodes=10, degree_dist=degree_dist, wealth_dist=wealth_dist)
    print("Id to wealth: ", sim.id_to_wealth)
    print("Id to income: ", sim.id_to_income)
    print("True order: ", sim.true_sorted_ids)
    print("Sorted order in sets: ", sim.sorted_nodes)
    print("Closest order: ", sim.best_rank_correlation_list)
    print("rank correl: ", sim.rank_correlation)
    print("gini: ", wealth_gini_all_nodes(sim.G, 'wealth'))
    print("degree dist: ", degree_dist(10))
    print("wealth dist: ", wealth_dist(10))
    sim.show()


if __name__ == "__main__":
    # run_one_sim()
    num_nodes_values = [10, 50, 100, 250, 500, 1000]
    mins = [np.array([2., 1000.])]
    alphas = [np.array([2., 2.])]
    correlations = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    noise_scales = [0, 0.1, 0.25, 0.5]
    sims_per_combination = 50

    run_simulations_varying_correlation(num_nodes_values, mins, alphas, correlations, noise_scales, sims_per_combination)





