import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import cvxpy as cp
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
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
    df["logSimulatedCPUs"] = np.log(df["SimulatedCPUs"])
    df["logNumFiles"] = np.log(df["NumFiles"])
    df["logConcurrency"] = np.log(df["Concurrency"])
    df["logTotalTime"] = np.log(df["TotalTime"])
    
    # =======================
    # 2. Feature Engineering
    # =======================
    # Create additional predictors:
    #   - logConcurrencySquared = (logConcurrency)^2
    #   - logSimCPUs_x_logConcurrency = logSimulatedCPUs * logConcurrency
    df["logConcurrencySquared"] = df["logConcurrency"] ** 2
    df["logSimCPUs_x_logConcurrency"] = df["logSimulatedCPUs"] * df["logConcurrency"]

    predictor_names = ["logSimulatedCPUs", "logNumFiles", "logConcurrency",
                       "logConcurrencySquared", "logSimCPUs_x_logConcurrency"]
    X = df[predictor_names]
    y = df["logTotalTime"]
    
    # =======================
    # 3. Cross-Validation for Ridge Regression
    # =======================
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Define a negative RMSE scorer (we want to minimize RMSE on the log scale)
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
    # Augment the design matrix so that we mimic Ridge regression while obtaining p-values.
    X_with_const = sm.add_constant(X)  # Adds column "const"
    # Separate the constant and predictors
    const_df = X_with_const[['const']]
    predictors_df = X_with_const.drop(columns="const")
    n_samples, n_features = predictors_df.shape  # n_features should be 5

    # Create augmented predictors: append sqrt(best_alpha)*I for predictors
    aug_predictors = pd.DataFrame(np.sqrt(best_alpha) * np.eye(n_features),
                                  columns=predictors_df.columns,
                                  index=["aug_" + col for col in predictors_df.columns])
    # Augmented constant: ones for the augmented rows (unpenalized)
    aug_const = pd.DataFrame(np.ones((n_features, 1)), columns=["const"],
                             index=aug_predictors.index)
    # Concatenate original data with augmented data
    X_augmented = pd.concat([X_with_const, pd.concat([aug_const, aug_predictors], axis=1)], axis=0)
    # Augmented response: original y values and zeros for augmented rows
    y_augmented = pd.concat([y, pd.Series(np.zeros(n_features), index=aug_predictors.index)])
    
    ridge_ols_model = sm.OLS(y_augmented, X_augmented).fit()
    print("\nFinal Ridge Regression Model (Augmented OLS) Summary:")
    print(ridge_ols_model.summary())
    
    # Extract interpretable coefficients. Our final model is:
    # logTotalTime = β₀ + β₁·logSimulatedCPUs + β₂·logNumFiles + β₃·logConcurrency +
    #                β₄·(logConcurrency)² + β₅·(logSimulatedCPUs × logConcurrency)
    beta0 = ridge_ols_model.params["const"]
    beta1 = ridge_ols_model.params["logSimulatedCPUs"]
    beta2 = ridge_ols_model.params["logNumFiles"]
    beta3 = ridge_ols_model.params["logConcurrency"]
    beta4 = ridge_ols_model.params["logConcurrencySquared"]
    beta5 = ridge_ols_model.params["logSimCPUs_x_logConcurrency"]

    # Print the final chosen model in plain text math format:
    final_model_str = (
        "Final Chosen Ridge Model:\n"
        "log(TotalTime) = {:.4f} + {:.4f} * log(SimulatedCPUs) + {:.4f} * log(NumFiles) + "
        "{:.4f} * log(Concurrency) + {:.4f} * (log(Concurrency))^2 + {:.4f} * (log(SimulatedCPUs) * log(Concurrency))"
    ).format(beta0, beta1, beta2, beta3, beta4, beta5)
    print("\n" + final_model_str)
    
    # =======================
    # 5. Convex Optimization for Optimal log(Concurrency)
    # =======================
    # For fixed values of logSimulatedCPUs (LSC) and logNumFiles (LNF), our model is:
    # f(x) = β₀ + β₁·LSC + β₂·LNF + (β₃ + β₅·LSC)*x + β₄*x², where x = log(Concurrency)
    # We constrain x >= 0 (so Concurrency >= 1).
    def optimize_log_concurrency(fixed_logSimulatedCPUs, fixed_logNumFiles):
        log_con_var = cp.Variable()  # Represents log(Concurrency)
        linear_coef = beta3 + beta5 * fixed_logSimulatedCPUs
        constant_term = beta0 + beta1 * fixed_logSimulatedCPUs + beta2 * fixed_logNumFiles
        objective = cp.Minimize(constant_term + linear_coef * log_con_var + beta4 * cp.square(log_con_var))
        constraints = [log_con_var >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return log_con_var.value

    # =======================
    # 6. Compute Optimal log(Concurrency) Surface over a Grid
    # =======================
    grid_points = 20
    # Use the original (unlogged) ranges for SimulatedCPUs and NumFiles.
    simulatedCPUs_vals = np.linspace(df["SimulatedCPUs"].min(), df["SimulatedCPUs"].max(), grid_points)
    numFiles_vals = np.linspace(df["NumFiles"].min(), df["NumFiles"].max(), grid_points)
    CPUs_grid, Files_grid = np.meshgrid(simulatedCPUs_vals, numFiles_vals)
    
    # Convert these to log space for our predictors.
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
    
    # Plot the optimal log(Concurrency) surface (grid already in log space)
    surface = ax.plot_surface(logSimCPUs_grid, logNumFiles_grid, optimal_logConcurrency_grid,
                              cmap="coolwarm", alpha=0.6)
    
    ax.set_xlabel("log(SimulatedCPUs)")
    ax.set_ylabel("log(NumFiles)")
    ax.set_zlabel("log(Concurrency)")
    ax.set_title("3D Plot (Logged):\nSimulatedCPUs, NumFiles, Concurrency\n(Color = log(TotalTime))\nOptimal log(Concurrency) Surface")
    
    # Create a custom legend entry for the surface using a valid color (red)
    from matplotlib.lines import Line2D
    custom_line = Line2D([0], [0], color="red", lw=4)
    ax.legend([custom_line], ["Optimal log(Concurrency) Surface"], loc="best")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
