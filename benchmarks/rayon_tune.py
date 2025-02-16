import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import cvxpy as cp
from matplotlib.colors import LinearSegmentedColormap

def main():
    # === 1. Load and Prepare Data ===
    # Expected CSV columns: SimulatedCPUs, NumFiles, Concurrency, TotalTime(ns)
    df = pd.read_csv("test_results.csv")
    # Rename TotalTime(ns) to TotalTime for convenience
    df.rename(columns={"TotalTime(ns)": "TotalTime"}, inplace=True)

    # Add a small constant to avoid log(0)
    epsilon = 1e-6
    for col in ['SimulatedCPUs', 'NumFiles', 'Concurrency', 'TotalTime']:
        df[col] = df[col] + epsilon

    # Log-transform all variables
    df['log_SimulatedCPUs'] = np.log(df['SimulatedCPUs'])
    df['log_NumFiles'] = np.log(df['NumFiles'])
    df['log_Concurrency'] = np.log(df['Concurrency'])
    df['log_TotalTime'] = np.log(df['TotalTime'])

    # === 2. Build a Quadratic Regression Model for log_TotalTime ===
    # Model: 
    # log_TotalTime = beta0 + beta1*log_SimulatedCPUs + beta2*log_NumFiles +
    #                 beta3*log_Concurrency + beta4*(log_Concurrency)^2 +
    #                 beta5*(log_SimulatedCPUs * log_Concurrency)
    df['log_Concurrency_sq'] = df['log_Concurrency'] ** 2
    df['log_CPUs_x_log_Conc'] = df['log_SimulatedCPUs'] * df['log_Concurrency']
    
    features = ['log_SimulatedCPUs', 'log_NumFiles', 'log_Concurrency',
                'log_Concurrency_sq', 'log_CPUs_x_log_Conc']
    X = df[features]
    X = sm.add_constant(X)
    y = df['log_TotalTime']

    model = sm.OLS(y, X).fit()
    print(model.summary())  # Displays the model summary with coefficients, p-values, and R^2

    # Extract coefficients for later use in optimization.
    beta0 = model.params["const"]
    beta1 = model.params["log_SimulatedCPUs"]
    beta2 = model.params["log_NumFiles"]
    beta3 = model.params["log_Concurrency"]
    beta4 = model.params["log_Concurrency_sq"]
    beta5 = model.params["log_CPUs_x_log_Conc"]

    # === 3. Set Up the Convex Optimization Problem ===
    # For fixed log_SimulatedCPUs (LSC) and log_NumFiles (LNF), the model becomes:
    # f(x) = beta0 + beta1*LSC + beta2*LNF + (beta3 + beta5*LSC)*x + beta4*x^2,
    # where x = log(Concurrency), and we require x >= 0 (i.e. Concurrency >= 1).
    def optimize_for_log_concurrency(LSC, LNF):
        x = cp.Variable()  # x = log(Concurrency)
        linear_coef = beta3 + beta5 * LSC
        constant_term = beta0 + beta1 * LSC + beta2 * LNF
        objective = cp.Minimize(constant_term + linear_coef * x + beta4 * cp.square(x))
        constraints = [x >= 0]  # Only physical bound: Concurrency cannot be < 1 (log(1)=0)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return x.value

    # === 4. Compute Optimal log(Concurrency) over a Grid for Visualization ===
    grid_points = 20
    # Use the original (unlogged) SimulatedCPUs and NumFiles ranges from the data
    SC_vals = np.linspace(df['SimulatedCPUs'].min(), df['SimulatedCPUs'].max(), grid_points)
    NF_vals = np.linspace(df['NumFiles'].min(), df['NumFiles'].max(), grid_points)
    SC_grid, NF_grid = np.meshgrid(SC_vals, NF_vals)
    opt_log_conc_grid = np.zeros_like(SC_grid)

    # For each grid point, compute optimal log(Concurrency)
    for i in range(grid_points):
        for j in range(grid_points):
            SC_val = SC_grid[i, j]
            NF_val = NF_grid[i, j]
            LSC = np.log(SC_val)
            LNF = np.log(NF_val)
            opt_log_conc_grid[i, j] = optimize_for_log_concurrency(LSC, LNF)
    
    # === 5. 3D Visualization ===
    # Axes: 
    #   x-axis: log(SimulatedCPUs)
    #   y-axis: log(NumFiles)
    #   z-axis: log(Concurrency)
    # Color represents log(TotalTime)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of observed data (logged values)
    sc = ax.scatter(df['log_SimulatedCPUs'], df['log_NumFiles'], df['log_Concurrency'],
                    c=df['log_TotalTime'], cmap='viridis', marker='o', alpha=0.8, s=50)

    # Convert the grid of SimulatedCPUs and NumFiles to log space for plotting
    log_SC_grid = np.log(SC_grid)
    log_NF_grid = np.log(NF_grid)
    # Plot the optimal function surface
    surf = ax.plot_surface(log_SC_grid, log_NF_grid, opt_log_conc_grid, cmap='coolwarm', alpha=0.6)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("log(TotalTime)")

    ax.set_xlabel("log(SimulatedCPUs)")
    ax.set_ylabel("log(NumFiles)")
    ax.set_zlabel("log(Concurrency)")
    ax.set_title("3D Plot (Logged):\nSimulatedCPUs, NumFiles, Concurrency\n(Color = log(TotalTime))\nOptimal Function Surface")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
