## Folder Structure

### **CPS**
Contains tools for processing CPS data based on the replication process defined in Hoffman (2020).
- `cps_loader.py` loads CPS data and processes it according to the replication process. Results are stored in the `Results` folder as samples by year and in `all_result.csv`. Before use, a the .xml codebook and .dat.gz data file need to be downloaded from the CPS website if you're trying to replicate anything!
- `calculators_data_calculators` includes `BaseCalculator` for common filtering, processing, saving, and parallelization steps. `IncomeStatisticsCalculator` extends it to calculate yearly basic statistics, and `GiniCalculator` adds Gini and Gini component calculations.
- `calculator_main.py` serves as the entry point to run calculators by specifying the desired calculator and parameters.
- `input-output_analysis.ipynb` contains initial exploration into using OECD input-output tables with CPS calculations, though this area is not currently developed.

### **Data**
Stores input data and processed outputs, including codebooks, swap values, and yearly processed CPS data.

### **Figures**
Stores output figures generated from the analysis.

### **IPUMS**
Includes code from past work for handling IPUMS data, forming state-by-industry networks using NetworkX, and analyzing and visualizing these results.

### **Network Utils**
Contains tools for network-based Gini calculations and sorting.
- `gini_from_netx.py` contains a class to calculate Gini component coefficients from networkX graphs. The Gini can be calculated either from a separated income and domination network or a single network with both attributes, with flexibly defined domination rules and attributes.
- `gini_calculation_from_network.py` defines functions to calculate Gini coefficients for different subsets of node pairs in directed graphs based on wealth and income attributes. This contains a lot of functions from previous work calcualting the gini out of sets of directed edges rather than out of the whole graph, and is now generally less useful than gini_from_netx.py
- `network_sort.py` processes and sorts directed graphs by collapsing strongly connected components into supernodes, then sorting supernodes and their nodes by topological order and net flow.

### **Results**
Stores results data from simulations, Gini calculations, and other statistical analyses.

### **Simulations**
Contains code for simulating networks and visualizing results.
- `simulations.py` simulates wealth and income dynamics on a directed network, where nodes represent individuals with wealth and income from user-defined distributions, and directed edges represent economic relationships influenced by wealth differentials. It includes support for multivariate Pareto distributions, robustness testing, and calculating Gini metrics for exploitation, patronage, rationing, and exclusion.
- `simulation_utils.py` provides tools for analyzing graphs during simulations, such as cycle analysis, node ranking, graph validation, and generating multivariate Pareto-distributed samples with specified correlations.
- `simulation_visualizations.ipynb` visualizes simulation results.

### **Tests**
Includes tests to verify replication of processed CPS data with Hoffman (2020) and to ensure the Gini calculator computes Gini coefficients and components correctly, matching known results.

---

## Usage Instructions

### **Run Decomposed Gini Calculations on CPS Data**
1. Use `cps_loader.py` to process CPS data.
2. Run `main_calculator.py` and specify `GiniCalculator` to calculate Gini metrics for each year.
3. Visualize results using `gini_visualizations.ipynb` (includes debugging graphs for income statistics).

### **Run Simulations**
1. Execute `simulations.py` with desired parameters.
2. Visualize simulation results using `simulation_visualizations.ipynb`.

### **Run Network Sorting**
1. Use `network_sort.py` to process and sort directed graphs.

### **Run Gini Calculations on NetworkX Graph(s)**
(0. Load data into networkX graph. This can be done in a process similarly to how it was executed in `ipums_data_extraction.ipynb` and `ipums_graph_analysis.ipynb` for IPUMS data)
1. Use `gini_from_netx.py`to calculate gini components. - Beware though that for large datasets this can easily overwhelm memory :( this is why CPS stateXindustry calculations and other calculations that involve fully connected components should be run using matrices in the GiniCalculator class!
