import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

def main():
    # =======================
    # 1. Load Data (for plotting ONLY)
    # =======================
    df = pd.read_csv("rayon_deletion_benchmark.csv", header=None,
                     names=["file_count", "time_in_seconds", "thread_pool_size", "batch_size"])
    df.rename(columns={"time_in_seconds": "TotalTime"}, inplace=True)
    epsilon = 1e-6
    for col in ["file_count", "TotalTime", "thread_pool_size", "batch_size"]:
        df[col] = df[col] + epsilon
    df["log_file_count"] = np.log(df["file_count"])
    df["logTotalTime"] = np.log(df["TotalTime"])
    df["log_thread_pool_size"] = np.log(df["thread_pool_size"])
    df["log_batch_size"] = np.log(df["batch_size"])

    # =======================
    # 2. Analytical Solution with SymPy (using PROVIDED coefficients, NO CONSTRAINTS)
    # =======================

    # Define symbolic variables with INTERPRETABLE names
    log_thread_pool_size, log_batch_size, log_file_count = sp.symbols('log_thread_pool_size log_batch_size log_file_count', real=True)
    thread_pool_size, batch_size, file_count = sp.symbols('thread_pool_size batch_size file_count', positive=True, real=True)

    # Coefficients (from the PROVIDED model... see other file)
    beta0 = -7.5276
    beta1 =  0.3250
    beta2 = -0.1292
    beta3 =  0.1885
    beta4 =  0.0884
    beta5 =  0.0078
    beta6 =  0.0563
    beta7 = -0.1019
    beta8 =  0.0009

    # Define the objective function symbolically (using PROVIDED coefficients)
    objective = (beta0 + beta1*log_thread_pool_size + beta2*log_batch_size + beta3*log_file_count +
                 beta4*(log_thread_pool_size**2) + beta5*(log_batch_size**2) + beta6*(log_file_count**2) +
                 beta7*log_file_count*log_thread_pool_size + beta8*log_file_count*log_thread_pool_size*log_batch_size)

    # Unconstrained Optimization
    d_objective_dlog_thread_pool_size = objective.diff(log_thread_pool_size)
    d_objective_dlog_batch_size = objective.diff(log_batch_size)
    unconstrained_solution = sp.solve([d_objective_dlog_thread_pool_size, d_objective_dlog_batch_size], (log_thread_pool_size, log_batch_size), dict=True)

    # Extract the solutions (handle potential multiple solutions or no solution)
    if unconstrained_solution:
        optimal_log_thread_pool_size = unconstrained_solution[0][log_thread_pool_size]
        optimal_log_batch_size = unconstrained_solution[0][log_batch_size]
    else:
        optimal_log_thread_pool_size = sp.nan
        optimal_log_batch_size = sp.nan


    # Convert back to original units
    optimal_thread_pool_size = sp.exp(optimal_log_thread_pool_size)
    optimal_batch_size = sp.exp(optimal_log_batch_size)

    print("\nOptimal thread_pool_size =")
    print(optimal_thread_pool_size)  # Plain text
    print("\nOptimal batch_size =")
    print(optimal_batch_size)         # Plain text

    # =======================
    # 3. 3D Visualization
    # =======================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot of the *original* data
    scatter = ax.scatter(df["log_file_count"], df["log_thread_pool_size"], df["log_batch_size"],
                         c=df["logTotalTime"], cmap="viridis", s=50, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("log(time_in_seconds)")

    # Compute the optimal decision curve using the ANALYTICAL solution:
    file_count_range = np.linspace(df["file_count"].min(), df["file_count"].max(), 150)
    optimal_log_thread_pool_analytical = []
    optimal_log_batch_analytical = []

    for fc in file_count_range:
        log_fc = np.log(fc)
        # Evaluate the expressions using lambdify for numerical evaluation
        log_T_func = sp.lambdify(log_file_count, optimal_log_thread_pool_size, modules=['numpy'])
        log_B_func = sp.lambdify(log_file_count, optimal_log_batch_size, modules=['numpy'])

        try:
            T_opt_analytical = log_T_func(log_fc)
            B_opt_analytical = log_B_func(log_fc)
            if isinstance(T_opt_analytical, np.ndarray):
                T_opt_analytical = T_opt_analytical.item() # Convert to scalar
            if isinstance(B_opt_analytical, np.ndarray):
                B_opt_analytical = B_opt_analytical.item() # Convert to scalar
            optimal_log_thread_pool_analytical.append(T_opt_analytical)
            optimal_log_batch_analytical.append(B_opt_analytical)

        except (TypeError, AttributeError) as e:
            optimal_log_thread_pool_analytical.append(np.nan)
            optimal_log_batch_analytical.append(np.nan)


    # Plot the optimal decision curve (from analytical solution)
    ax.plot(np.log(file_count_range), optimal_log_thread_pool_analytical, optimal_log_batch_analytical,
            color="red", linewidth=3, label="Optimal Decision Curve (Analytical)")

    ax.set_xlabel("log(file_count)")
    ax.set_ylabel("log(thread_pool_size)")
    ax.set_zlabel("log(batch_size)")
    ax.set_title("3D Plot (Logged): file_count, thread_pool_size, batch_size (Color = log(time_in_seconds))")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
