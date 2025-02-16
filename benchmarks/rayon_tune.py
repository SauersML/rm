import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import cvxpy as cp
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    # =======================
    # 1. Load and Prepare Data
    # =======================
    # Expected CSV columns: SimulatedCPUs, NumFiles, Concurrency, TotalTime(ns)
    df = pd.read_csv("test_results.csv")
    df.rename(columns={"TotalTime(ns)": "TotalTime"}, inplace=True)
    
    # Add a small constant to avoid log(0)
    epsilon = 1e-6
    for col in ['SimulatedCPUs', 'NumFiles', 'Concurrency', 'TotalTime']:
        df[col] = df[col] + epsilon

    # Log-transform all variables with interpretable names
    df['logSimulatedCPUs'] = np.log(df['SimulatedCPUs'])
    df['logNumFiles'] = np.log(df['NumFiles'])
    df['logConcurrency'] = np.log(df['Concurrency'])
    df['logTotalTime'] = np.log(df['TotalTime'])
    
    # =======================
    # 2. Feature Engineering
    # =======================
    # We use the following predictors:
    #   - logSimulatedCPUs
    #   - logNumFiles
    #   - logConcurrency
    #   - logConcurrencySquared: (logConcurrency)^2
    #   - logSimCPUs_x_logConcurrency: logSimulatedCPUs * logConcurrency
    df['logConcurrencySquared'] = df['logConcurrency'] ** 2
    df['logSimCPUs_x_logConcurrency'] = df['logSimulatedCPUs'] * df['logConcurrency']

    predictors = ["logSimulatedCPUs", "logNumFiles", "logConcurrency",
                  "logConcurrencySquared", "logSimCPUs_x_logConcurrency"]
    X = df[predictors]
    y = df['logTotalTime']
    
    # =======================
    # 3. Cross-Validation for Ridge Regression
    # =======================
    # We use 5-fold CV and GridSearchCV to tune Ridge's alpha.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Define a negative RMSE scorer (since GridSearchCV maximizes score)
    def neg_rmse(estimator, X_val, y_val):
        predictions = estimator.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        return -rmse

    ridge = Ridge(fit_intercept=True)
    alphas = np.logspace(-2, 2, 50)
    param_grid = {'alpha': alphas}
    ridge_grid = GridSearchCV(ridge, param_grid, cv=cv, scoring=neg_rmse)
    ridge_grid.fit(X, y)
    best_alpha = ridge_grid.best_params_['alpha']
    best_ridge_rmse = -ridge_grid.best_score_
    print("Ridge CV Results:")
    print(f"  Best alpha: {best_alpha:.4f}")
    print(f"  Best CV RMSE (log scale): {best_ridge_rmse:.4f}")
    
    # =======================
    # 4. Refit Ridge Regression via Augmented OLS to Obtain p-values
    # =======================
    # We mimic Ridge regression using data augmentation so that p-values are available.
    # (We do not penalize the intercept.)
    X_with_const = sm.add_constant(X)  # This returns a DataFrame with columns: "const", then predictors.
    # Separate the constant and predictors:
    const_col = X_with_const[['const']]
    X_predictors = X_with_const.drop(columns="const")
    n_samples, n_features = X_predictors.shape  # n_features should be 5

    # Create augmented predictors DataFrame for non-constant terms:
    # Each augmented row is sqrt(best_alpha)*I (with the same column names as X_predictors)
    aug_predictors = pd.DataFrame(np.sqrt(best_alpha) * np.eye(n_features),
                                  columns=X_predictors.columns,
                                  index=["aug_" + col for col in X_predictors.columns])
    # Create augmented constant: ones for each augmented row (not penalized)
    aug_const = pd.DataFrame(np.ones((n_features, 1)), columns=["const"],
                             index=aug_predictors.index)
    aug_df = pd.concat([aug_const, aug_predictors], axis=1)
    
    # Final augmented design matrix: concatenate original and augmented data
    X_final = pd.concat([X_with_const, aug_df], axis=0)
    # Augmented response: original y and zeros for the augmented rows
    y_final = pd.concat([y, pd.Series(np.zeros(n_features), index=aug_df.index)])
    
    ridge_augmented_model = sm.OLS(y_final, X_final).fit()
    print("\nFinal Ridge Regression Model (Augmented OLS) Summary:")
    print(ridge_augmented_model.summary())
    
    # For subsequent optimization, extract the coefficients for our interpretable predictors:
    # Our model is: 
    # logTotalTime = beta0 + beta1*logSimulatedCPUs + beta2*logNumFiles +
    #                beta3*logConcurrency + beta4*logConcurrencySquared +
    #                beta5*logSimCPUs_x_logConcurrency
    beta0 = ridge_augmented_model.params["const"]
    beta1 = ridge_augmented_model.params["logSimulatedCPUs"]
    beta2 = ridge_augmented_model.params["logNumFiles"]
    beta3 = ridge_augmented_model.params["logConcurrency"]
    beta4 = ridge_augmented_model.params["logConcurrencySquared"]
    beta5 = ridge_augmented_model.params["logSimCPUs_x_logConcurrency"]

    # =======================
    # 5. Convex Optimization for Optimal log(Concurrency)
    # =======================
    # For fixed values of logSimulatedCPUs (LSC) and logNumFiles (LNF), our model becomes:
    # f(logCon) = beta0 + beta1*LSC + beta2*LNF + (beta3 + beta5*LSC)*logCon + beta4*(logCon)^2,
    # where logCon = log(Concurrency) and we require logCon >= 0 (i.e. Concurrency >= 1).
    def optimize_log_concurrency(fixed_logSimCPUs, fixed_logNumFiles):
        log_con_var = cp.Variable()  # Represents log(Concurrency)
        linear_term = beta3 + beta5 * fixed_logSimCPUs
        constant_term = beta0 + beta1 * fixed_logSimCPUs + beta2 * fixed_logNumFiles
        objective = cp.Minimize(constant_term + linear_term * log_con_var + beta4 * cp.square(log_con_var))
        constraints = [log_con_var >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return log_con_var.value

    # =======================
    # 6. Compute Optimal log(Concurrency) Surface over a Grid
    # =======================
    grid_points = 20
    # We'll use the original (unlogged) ranges for SimulatedCPUs and NumFiles,
    # then convert to log-scale.
    simulatedCPUs_range = np.linspace(df["SimulatedCPUs"].min(), df["SimulatedCPUs"].max(), grid_points)
    numFiles_range = np.linspace(df["NumFiles"].min(), df["NumFiles"].max(), grid_points)
    CPUs_grid, Files_grid = np.meshgrid(simulatedCPUs_range, numFiles_range)
    
    # Build grids in log space for our predictors:
    logSimCPUs_grid = np.log(CPUs_grid)
    logNumFiles_grid = np.log(Files_grid)
    
    optimal_logConcurrency_grid = np.zeros_like(logSimCPUs_grid)
    
    for i in range(grid_points):
        for j in range(grid_points):
            fixed_logSimCPUs = logSimCPUs_grid[i, j]
            fixed_logNumFiles = logNumFiles_grid[i, j]
            optimal_logConcurrency_grid[i, j] = optimize_log_concurrency(fixed_logSimCPUs, fixed_logNumFiles)
    
    # =======================
    # 7. 3D Visualization
    # =======================
    # Axes (all in log space):
    #   x-axis: logSimulatedCPUs
    #   y-axis: logNumFiles
    #   z-axis: log(Concurrency)
    # Observed data points are colored by logTotalTime.
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    
    scatter = ax.scatter(df["logSimulatedCPUs"], df["logNumFiles"], df["logConcurrency"],
                         c=df["logTotalTime"], cmap="viridis", s=50, alpha=0.8)
    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    colorbar.set_label("log(TotalTime)")
    
    # Plot the optimal surface (which should be a plane if the model is linear in log(Concurrency))
    surface = ax.plot_surface(logSimCPUs_grid, logNumFiles_grid, optimal_logConcurrency_grid,
                              cmap="coolwarm", alpha=0.6)
    
    ax.set_xlabel("log(SimulatedCPUs)")
    ax.set_ylabel("log(NumFiles)")
    ax.set_zlabel("log(Concurrency)")
    ax.set_title("3D Plot (Logged):\nSimulatedCPUs, NumFiles, Concurrency\n(Color = log(TotalTime))\nOptimal log(Concurrency) Surface")
    
    # Create a custom legend entry for the surface using a valid color (e.g., red)
    from matplotlib.lines import Line2D
    custom_line = Line2D([0], [0], color="red", lw=4)
    ax.legend([custom_line], ["Optimal log(Concurrency) Surface"], loc="best")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
