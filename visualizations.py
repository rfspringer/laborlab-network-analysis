import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def heatmap():
    df = pd.read_csv('results/simulation_results_partially_connected_no_noise.csv')

    pivot_df = df.pivot_table(index='num_nodes', columns='p', values='rank_correl')

    pivot_df = pivot_df[::-1]

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=0.5)
    plt.title('Rank Correlation Heatmap- Partially Connected Graphs, No Noise')
    plt.xlabel('prob_connection')
    plt.ylabel('num_nodes')
    plt.show()

def bargraphs():
    df = pd.read_csv('results/simulation_results_fully_connected_noise.csv')

    # Get unique num_nodes values
    num_nodes_values = sorted(df['num_nodes'].unique())

    # Calculate maximum and minimum values of rank_correl for each num_nodes value
    max_values = df.groupby('num_nodes')['rank_correl'].max()
    min_values = df.groupby('num_nodes')['rank_correl'].min()
    errors = [max_values - min_values, np.zeros_like(min_values)]  # Error bars for the range between min and max

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(num_nodes_values, (max_values + min_values) / 2, yerr=errors, fmt='o', color='blue', capsize=5, linewidth=2)
    plt.xlabel('num_nodes')
    plt.ylabel('Rank Correlation')
    plt.title('Rank Correlation- Fully Connected Graphs with Noise')
    plt.ylim(0.8, 1)
    plt.grid(axis='y')
    plt.show()


bargraphs()
heatmap()