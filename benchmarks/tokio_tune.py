import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from scipy.optimize import minimize

def main():
    # === 1. Load Data ===
    # Expected CSV columns: SimulatedCPUs, NumFiles, Concurrency, TotalTime(ns)
    df = pd.read_csv("test_results.csv")
    df.rename(columns={"TotalTime(ns)": "TotalTime"}, inplace=True)
    
    # === 2. Data Preprocessing ===
    # Add a small constant to avoid log(0)
    epsilon = 1e-6
    for col in ['SimulatedCPUs', 'NumFiles', 'Concurrency', 'TotalTime']:
        df[col] = df[col] + epsilon

    # Log-transform all variables
    df['log_SimulatedCPUs'] = np.log(df['SimulatedCPUs'])
    df['log_NumFiles'] = np.log(df['NumFiles'])
    df['log_Concurrency'] = np.log(df['Concurrency'])
    df['log_TotalTime'] = np.log(df['TotalTime'])
    
    # === 3. Feature Engineering (Interaction Terms) ===
    df['log_CPUs_x_log_Files'] = df['log_SimulatedCPUs'] * df['log_NumFiles']
    df['log_CPUs_x_log_Conc'] = df['log_SimulatedCPUs'] * df['log_Concurrency']
    df['log_Files_x_log_Conc'] = df['log_NumFiles'] * df['log_Concurrency']
    
    # === 4. Model Training using Statsmodels OLS ===
    features = ['log_SimulatedCPUs', 'log_NumFiles', 'log_Concurrency',
                'log_CPUs_x_log_Files', 'log_CPUs_x_log_Conc', 'log_Files_x_log_Conc']
    X = df[features]
    X = sm.add_constant(X)
    y = df['log_TotalTime']
    
    model = sm.OLS(y, X).fit()
    print(model.summary())  # Displays coefficients, p-values, R^2, etc.
    
    # === 5. Define Prediction and Optimization Functions ===
    def predict_log_total_time(SimulatedCPUs, NumFiles, Concurrency):
        # Compute logged inputs
        log_SimCPUs = np.log(SimulatedCPUs + epsilon)
        log_NumFiles = np.log(NumFiles + epsilon)
        log_Conc = np.log(Concurrency + epsilon)
        # Build a DataFrame for a single observation
        data = {
            'const': [1.0],
            'log_SimulatedCPUs': [log_SimCPUs],
            'log_NumFiles': [log_NumFiles],
            'log_Concurrency': [log_Conc],
            'log_CPUs_x_log_Files': [log_SimCPUs * log_NumFiles],
            'log_CPUs_x_log_Conc': [log_SimCPUs * log_Conc],
            'log_Files_x_log_Conc': [log_NumFiles * log_Conc]
        }
        df_input = pd.DataFrame(data)
        return model.predict(df_input)[0]
    
    def optimal_concurrency(SimulatedCPUs, NumFiles, Concurrency_min=1, Concurrency_max=None):
        # Optimize over log(Concurrency) for numerical stability.
        if Concurrency_max is None:
            Concurrency_max = 2 * SimulatedCPUs  # default upper bound
        
        def objective(log_Conc):
            # log_Conc is a one-element array; convert to actual Concurrency
            Conc = np.exp(log_Conc[0])
            return predict_log_total_time(SimulatedCPUs, NumFiles, Conc)
        
        bounds = [(np.log(Concurrency_min), np.log(Concurrency_max))]
        initial_guess = [np.log(max(Concurrency_min, min(SimulatedCPUs, Concurrency_max)))]
        result = minimize(objective, x0=initial_guess, bounds=bounds, method='L-BFGS-B')
        opt_log_Conc = result.x[0]
        opt_Conc = np.exp(opt_log_Conc)
        opt_log_TotalTime = predict_log_total_time(SimulatedCPUs, NumFiles, opt_Conc)
        return opt_Conc, opt_log_TotalTime
    
    # === 6. Example Optimization for Fixed SimulatedCPUs and NumFiles ===
    SimulatedCPUs_fixed = 16
    NumFiles_fixed = 1024
    opt_conc, opt_log_total_time = optimal_concurrency(SimulatedCPUs_fixed, NumFiles_fixed)
    opt_total_time = np.exp(opt_log_total_time)
    print(f"\nFor {SimulatedCPUs_fixed} CPUs and {NumFiles_fixed} Files:")
    print(f"Optimal Concurrency: {opt_conc:.2f}")
    print(f"Minimum Total Time: {opt_total_time:.2f} ns")
    
    # === 7. 3D Visualization ===
    # Axes: x = log(SimulatedCPUs), y = log(NumFiles), z = log(Concurrency)
    # Color of points: log(TotalTime)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot observed data points (logged values)
    sc = ax.scatter(df['log_SimulatedCPUs'], df['log_NumFiles'], df['log_Concurrency'],
                    c=df['log_TotalTime'], cmap='viridis', marker='o', alpha=0.8, s=50)
    
    # Compute grid for optimal function surface over SimulatedCPUs and NumFiles
    grid_points = 20
    SC_vals = np.linspace(df['SimulatedCPUs'].min(), df['SimulatedCPUs'].max(), grid_points)
    NF_vals = np.linspace(df['NumFiles'].min(), df['NumFiles'].max(), grid_points)
    SC_grid, NF_grid = np.meshgrid(SC_vals, NF_vals)
    opt_log_conc_grid = np.zeros_like(SC_grid)
    
    # For each grid point, compute the optimal log(Concurrency)
    for i in range(grid_points):
        for j in range(grid_points):
            SC_val = SC_grid[i, j]
            NF_val = NF_grid[i, j]
            opt_conc_val, _ = optimal_concurrency(SC_val, NF_val, Concurrency_min=1, Concurrency_max=2*SC_val)
            opt_log_conc_grid[i, j] = np.log(opt_conc_val + epsilon)
    
    # Convert SimulatedCPUs and NumFiles to log space for plotting
    log_SC_grid = np.log(SC_grid)
    log_NF_grid = np.log(NF_grid)
    
    # Plot the optimal function surface (surface of optimal log(Concurrency))
    surf = ax.plot_surface(log_SC_grid, log_NF_grid, opt_log_conc_grid, cmap='coolwarm', alpha=0.6)
    
    # Add colorbar for the observed log(TotalTime) scatter
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
