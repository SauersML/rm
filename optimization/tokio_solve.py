import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def main():
    # =========================================================================
    # Analytical Solution Derivation using SymPy
    # =========================================================================

    # Define symbolic variables with clear names
    log_simulated_cpus, log_num_files, log_concurrency = sp.symbols('log_simulated_cpus log_num_files log_concurrency', real=True)
    simulated_cpus, num_files, concurrency = sp.symbols('simulated_cpus num_files concurrency', positive=True, real=True)

    # Coefficients
    beta0 = 11.7759
    beta1 = -0.6425
    beta2 = 0.8490
    beta3 = -0.0665
    beta4 = 0.1106
    beta5 = 0.0101
    beta6 = 0.0105
    beta7 = 0.0116
    beta8 = -0.0029

    # Define the objective function symbolically
    objective = (beta0 + beta1 * log_simulated_cpus + beta2 * log_num_files + beta3 * log_concurrency +
                 beta4 * (log_simulated_cpus**2) + beta5 * (log_num_files**2) + beta6 * (log_concurrency**2) +
                 beta7 * log_num_files * log_concurrency + beta8 * log_simulated_cpus * log_num_files * log_concurrency)

    # 1. Unconstrained Optimization
    d_objective_dlog_concurrency = objective.diff(log_concurrency)
    unconstrained_solution = sp.solve(d_objective_dlog_concurrency, log_concurrency, dict=True)

    if unconstrained_solution:
        optimal_log_concurrency = unconstrained_solution[0][log_concurrency]
    else:
        optimal_log_concurrency = sp.nan

    # 4. Convert back to original units (for display)
    optimal_concurrency = sp.exp(optimal_log_concurrency)

    print("\nOptimal log(Concurrency):")
    print(optimal_log_concurrency)
    print("\nOptimal Concurrency:")
    print(optimal_concurrency)  # Plain text

    # =======================
    # Data Loading and Plotting (for comparison with analytical solution)
    # =======================
    df = pd.read_csv("test_results.csv")
    df.rename(columns={"TotalTime(ns)": "TotalTime"}, inplace=True)
    epsilon = 1e-6
    for col in ['SimulatedCPUs', 'NumFiles', 'Concurrency', 'TotalTime']:
        df[col] = df[col] + epsilon
    df["logSimulatedCPUs"] = np.log(df["SimulatedCPUs"])
    df["logNumFiles"] = np.log(df["NumFiles"])
    df["logConcurrency"] = np.log(df["Concurrency"])
    df["logTotalTime"] = np.log(df["TotalTime"])

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of original data
    scatter = ax.scatter(df["logSimulatedCPUs"], df["logNumFiles"], df["logConcurrency"],
                         c=df["logTotalTime"], cmap='viridis', s=50, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('log(TotalTime)')

    # Generate points for the optimal Concurrency surface
    num_points = 50
    simulated_cpus_range = np.linspace(df["SimulatedCPUs"].min(), df["SimulatedCPUs"].max(), num_points)
    num_files_range = np.linspace(df["NumFiles"].min(), df["NumFiles"].max(), num_points)
    simulated_cpus_grid, num_files_grid = np.meshgrid(simulated_cpus_range, num_files_range)
    log_simulated_cpus_grid = np.log(simulated_cpus_grid)
    log_num_files_grid = np.log(num_files_grid)
    optimal_concurrency_grid = np.zeros_like(simulated_cpus_grid, dtype=float)

    # Use lambdify to create numerical functions from the symbolic expressions
    optimal_log_concurrency_func = sp.lambdify((log_simulated_cpus, log_num_files), optimal_log_concurrency, modules='numpy')

    for i in range(num_points):
        for j in range(num_points):
            log_C_val = log_simulated_cpus_grid[i, j]
            log_F_val = log_num_files_grid[i, j]

            # Evaluate the optimal log(Concurrency)
            try:
                opt_log_concurrency = optimal_log_concurrency_func(log_C_val, log_F_val)
                if isinstance(opt_log_concurrency, np.ndarray):
                    opt_log_concurrency = opt_log_concurrency.item()
                optimal_concurrency_grid[i, j] = opt_log_concurrency
            except (TypeError, AttributeError) as e:
                optimal_concurrency_grid[i, j] = np.nan

    # Plot the optimal surface
    ax.plot_surface(log_simulated_cpus_grid, log_num_files_grid, optimal_concurrency_grid,
                    cmap='coolwarm', alpha=0.6)
    
    ax.set_xlabel('log(SimulatedCPUs)')
    ax.set_ylabel('log(NumFiles)')
    ax.set_zlabel('log(Concurrency)')
    ax.set_title('Optimal Concurrency Surface and Data')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
