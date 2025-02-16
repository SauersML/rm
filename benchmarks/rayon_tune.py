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
    # The CSV (with no header) has columns (in order):
    # file_count, time_in_seconds, thread_pool_size, batch_size
    df = pd.read_csv("rayon_deletion_benchmark.csv", header=None,
                     names=["file_count", "time_in_seconds", "thread_pool_size", "batch_size"])
    # Rename "time_in_seconds" to "TotalTime" for clarity.
    df.rename(columns={"time_in_seconds": "TotalTime"}, inplace=True)
    
    # Add a small constant to avoid log(0)
    epsilon = 1e-6
    for col in ["file_count", "TotalTime", "thread_pool_size", "batch_size"]:
        df[col] = df[col] + epsilon

    # Log-transform all variables with interpretable names
    df["log_file_count"] = np.log(df["file_count"])
    df["logTotalTime"] = np.log(df["TotalTime"])
    df["log_thread_pool_size"] = np.log(df["thread_pool_size"])
    df["log_batch_size"] = np.log(df["batch_size"])
    
    # =======================
    # 2. Feature Engineering
    # =======================
    # We model log(TotalTime) as a function of:
    #   - log(thread_pool_size)
    #   - log(batch_size)
    #   - log(file_count)  (exogenous)
    #   - (log(thread_pool_size))^2
    #   - (log(batch_size))^2
    #   - (log(thread_pool_size) * log(batch_size))
    df["log_thread_pool_size_sq"] = df["log_thread_pool_size"] ** 2
    df["log_batch_size_sq"] = df["log_batch_size"] ** 2
    df["log_tp_bs_interaction"] = df["log_thread_pool_size"] * df["log_batch_size"]
    
    predictors = ["log_thread_pool_size", "log_batch_size", "log_file_count",
                  "log_thread_pool_size_sq", "log_batch_size_sq", "log_tp_bs_interaction"]
    X = df[predictors]
    y = df["logTotalTime"]
    
    # =======================
    # 3. Cross-Validation for Ridge Regression
    # =======================
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    def neg_rmse(estimator, X_val, y_val):
        predictions = estimator.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        return -rmse

    ridge = Ridge(fit_intercept=True)
    alphas = np.logspace(-2, 2, 50)
    param_grid = {"alpha": alphas}
    ridge_grid = GridSearchCV(ridge, param_grid, cv=cv, scoring=neg_rmse)
    ridge_grid.fit(X, y)
    best_alpha = ridge_grid.best_params_["alpha"]
    best_ridge_rmse = -ridge_grid.best_score_
    print("Ridge CV Results:")
    print(f"  Best alpha: {best_alpha:.4f}")
    print(f"  Best CV RMSE (log scale): {best_ridge_rmse:.4f}")
    
    # =======================
    # 4. Refit Ridge via Augmented OLS to Obtain p-values
    # =======================
    # We augment the design matrix (penalizing only predictors, not the intercept)
    X_with_const = sm.add_constant(X)  # adds a "const" column
    # Separate the constant and predictors
    const_df = X_with_const[["const"]]
    predictors_df = X_with_const.drop(columns="const")
    n_samples, n_features = predictors_df.shape  # should be 6

    # Create augmented predictors: append sqrt(best_alpha)*I for each predictor (with same column names)
    aug_predictors = pd.DataFrame(np.sqrt(best_alpha) * np.eye(n_features),
                                  columns=predictors_df.columns,
                                  index=["aug_" + col for col in predictors_df.columns])
    # Create augmented constant: ones for each augmented row (unpenalized)
    aug_const = pd.DataFrame(np.ones((n_features, 1)), columns=["const"],
                             index=aug_predictors.index)
    X_augmented = pd.concat([X_with_const, pd.concat([aug_const, aug_predictors], axis=1)], axis=0)
    y_augmented = pd.concat([y, pd.Series(np.zeros(n_features), index=aug_predictors.index)])
    
    ridge_ols_model = sm.OLS(y_augmented, X_augmented).fit()
    print("\nFinal Ridge Regression Model (Augmented OLS) Summary:")
    print(ridge_ols_model.summary())
    
    # Extract coefficients (interpretable names)
    # Final model:
    # log(time_in_seconds) = β₀ + β₁*log(thread_pool_size) + β₂*log(batch_size) +
    #                        β₃*log(file_count) + β₄*(log(thread_pool_size))^2 + β₅*(log(batch_size))^2 +
    #                        β₆*(log(thread_pool_size) * log(batch_size))
    beta0 = ridge_ols_model.params["const"]
    beta1 = ridge_ols_model.params["log_thread_pool_size"]
    beta2 = ridge_ols_model.params["log_batch_size"]
    beta3 = ridge_ols_model.params["log_file_count"]
    beta4 = ridge_ols_model.params["log_thread_pool_size_sq"]
    beta5 = ridge_ols_model.params["log_batch_size_sq"]
    beta6 = ridge_ols_model.params["log_tp_bs_interaction"]

    final_model_str = (
        "Final Chosen Ridge Model:\n"
        "log(time_in_seconds) = {:.4f} + {:.4f} * log(thread_pool_size) + {:.4f} * log(batch_size) + "
        "{:.4f} * log(file_count) + {:.4f} * (log(thread_pool_size))^2 + {:.4f} * (log(batch_size))^2 + "
        "{:.4f} * (log(thread_pool_size) * log(batch_size))"
    ).format(beta0, beta1, beta2, beta3, beta4, beta5, beta6)
    print("\n" + final_model_str)
    
    # =======================
    # 5. Convex Optimization for Optimal Decision Variables
    # =======================
    # We can modify two variables: thread_pool_size and batch_size.
    # Let T = log(thread_pool_size) and B = log(batch_size), and let F = log(file_count) be fixed.
    # Our model is:
    # log(time_in_seconds) = β₀ + β₁*T + β₂*B + β₃*F + β₄*T² + β₅*B² + β₆*(T*B)
    # For a fixed F, define:
    # f(T, B) = (β₀ + β₃*F) + β₁*T + β₂*B + β₄*T² + β₅*B² + β₆*(T*B)
    # We minimize f(T, B) subject to T >= 0 and B >= 0.
    def optimize_decision_variables(fixed_logFileCount):
        T = cp.Variable()  # T = log(thread_pool_size)
        B = cp.Variable()  # B = log(batch_size)
        constant_part = beta0 + beta3 * fixed_logFileCount
        linear_part = beta1 * T + beta2 * B
        Q = np.array([[beta4, beta6 / 2.0],
                      [beta6 / 2.0, beta5]])
        quad_part = cp.quad_form(cp.vstack([T, B]), Q)
        objective = cp.Minimize(constant_part + linear_part + quad_part)
        constraints = [T >= 0, B >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return T.value, B.value, prob.value

    # For example, fix file_count = 100, so F = log(100)
    fixed_file_count = 100
    fixed_logFileCount = np.log(fixed_file_count)
    opt_log_T, opt_log_B, optimal_obj = optimize_decision_variables(fixed_logFileCount)
    opt_thread_pool_size = np.exp(opt_log_T)
    opt_batch_size = np.exp(opt_log_B)
    predicted_time = np.exp(optimal_obj)
    
    print("\nFor file_count = {}:".format(fixed_file_count))
    print("Optimal thread_pool_size (predicted, original scale): {:.2f}".format(opt_thread_pool_size))
    print("Optimal batch_size (predicted, original scale): {:.2f}".format(opt_batch_size))
    print("Predicted minimum time (seconds): {:.6f}".format(predicted_time))
    
    # =======================
    # 6. 3D Visualization: Optimal Surface for Decision Variables
    # =======================
    # The 3D plot will use:
    #   x-axis: log(file_count)
    #   y-axis: log(thread_pool_size)
    #   z-axis: log(batch_size)
    # Color of the points: log(time_in_seconds)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    
    # Scatter plot: axes are the three independent variables (in log scale)
    scatter = ax.scatter(df["log_file_count"], df["log_thread_pool_size"], df["log_batch_size"],
                         c=df["logTotalTime"], cmap="viridis", s=50, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("log(time_in_seconds)")
    
    # Compute the optimal decision curve:
    # For a range of file_count values, compute the optimal thread_pool_size and batch_size.
    file_count_range = np.linspace(df["file_count"].min(), df["file_count"].max(), 50)
    optimal_log_thread_pool = []
    optimal_log_batch = []
    log_file_count_vals = np.log(file_count_range)
    for fc in file_count_range:
        T_opt, B_opt, _ = optimize_decision_variables(np.log(fc))
        optimal_log_thread_pool.append(T_opt)
        optimal_log_batch.append(B_opt)
    
    # Plot the optimal decision curve in the 3D space:
    ax.plot(np.log(file_count_range), optimal_log_thread_pool, optimal_log_batch,
            color="red", linewidth=3, label="Optimal Decision Curve")
    
    ax.set_xlabel("log(file_count)")
    ax.set_ylabel("log(thread_pool_size)")
    ax.set_zlabel("log(batch_size)")
    ax.set_title("3D Plot (Logged):\nfile_count, thread_pool_size, batch_size\n(Color = log(time_in_seconds))\nOptimal Decision Curve")
    ax.legend(loc="best")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
