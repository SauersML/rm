import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def main():
    # =======================
    # 1. Load and Prepare Data
    # =======================
    # CSV columns: file_count, time_in_seconds, thread_pool_size, batch_size
    df = pd.read_csv("rayon_deletion_benchmark.csv", header=None,
                     names=["file_count", "time_in_seconds", "thread_pool_size", "batch_size"])
    # Rename for clarity
    df.rename(columns={"time_in_seconds": "TotalTime"}, inplace=True)
    
    # Avoid issues with log(0)
    epsilon = 1e-6
    for col in ["file_count", "TotalTime", "thread_pool_size", "batch_size"]:
        df[col] = df[col] + epsilon

    # Compute logged variables
    df["logTotalTime"] = np.log(df["TotalTime"])
    df["log_thread_pool_size"] = np.log(df["thread_pool_size"])  # T
    df["log_batch_size"] = np.log(df["batch_size"])                # B
    df["log_file_count"] = np.log(df["file_count"])                # F

    # =======================
    # 2. Feature Engineering for the Model
    # =======================
    # Create squared terms
    df["log_thread_pool_size_sq"] = df["log_thread_pool_size"] ** 2
    df["log_batch_size_sq"] = df["log_batch_size"] ** 2
    df["log_file_count_sq"] = df["log_file_count"] ** 2

    # Create interaction terms
    # (Note: The log(T)*log(B) term has been REMOVED as requested.)
    df["log_fc_tp_interaction"] = df["log_file_count"] * df["log_thread_pool_size"]        # log(F)*log(T)
    df["log_fc_tp_bs_interaction"] = df["log_file_count"] * df["log_thread_pool_size"] * df["log_batch_size"]  # log(F)*log(T)*log(B)

    # Predictors list per the proposed model (without log(T)*log(B))
    predictors = [
        "log_thread_pool_size",       # T
        "log_batch_size",             # B
        "log_file_count",             # F
        "log_thread_pool_size_sq",    # T²
        "log_batch_size_sq",          # B²
        "log_file_count_sq",          # F²
        "log_fc_tp_interaction",      # F*T
        "log_fc_tp_bs_interaction"    # F*T*B
    ]
    X = df[predictors]
    y = df["logTotalTime"]

    # =======================
    # 3. Ridge Regression with Cross-Validation
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
    # 4. Refit Ridge via Augmented OLS (to retrieve coefficients)
    # =======================
    X_with_const = sm.add_constant(X)
    predictors_df = X_with_const.drop(columns="const")
    n_samples, n_features = predictors_df.shape

    # Augment predictors for ridge penalty
    aug_predictors = pd.DataFrame(
        np.sqrt(best_alpha) * np.eye(n_features),
        columns=predictors_df.columns,
        index=["aug_" + col for col in predictors_df.columns]
    )
    # Augment constant with zeros
    aug_const = pd.DataFrame(
        np.zeros((n_features, 1)),
        columns=["const"],
        index=aug_predictors.index
    )
    # Combine original and augmented data
    X_augmented = pd.concat([X_with_const, pd.concat([aug_const, aug_predictors], axis=1)], axis=0)
    y_augmented = pd.concat([y, pd.Series(np.zeros(n_features), index=aug_predictors.index)])

    ridge_ols_model = sm.OLS(y_augmented, X_augmented).fit()
    print("\nFinal Ridge Regression Model (Augmented OLS) Summary:")
    print(ridge_ols_model.summary())

    # Extract coefficients from the fitted model
    beta0 = ridge_ols_model.params["const"]
    beta1 = ridge_ols_model.params["log_thread_pool_size"]
    beta2 = ridge_ols_model.params["log_batch_size"]
    beta3 = ridge_ols_model.params["log_file_count"]
    beta4 = ridge_ols_model.params["log_thread_pool_size_sq"]
    beta5 = ridge_ols_model.params["log_batch_size_sq"]
    beta6 = ridge_ols_model.params["log_file_count_sq"]
    beta7 = ridge_ols_model.params["log_fc_tp_interaction"]
    beta8 = ridge_ols_model.params["log_fc_tp_bs_interaction"]

    final_model_str = (
        "Final Chosen Ridge Model:\n"
        "log(time_in_seconds) = {:.4f} + {:.4f}*log(T) + {:.4f}*log(B) + {:.4f}*log(F) +\n"
        "  {:.4f}*log(T)^2 + {:.4f}*log(B)^2 + {:.4f}*log(F)^2 +\n"
        "  {:.4f}*log(F)*log(T) + {:.4f}*log(F)*log(T)*log(B)"
    ).format(beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7, beta8)
    print("\n" + final_model_str)

    # =======================
    # 5. 3D Visualization of the Data and Model Predictions
    # =======================
    # Here we plot the three independent variables (in log-scale):
    #   x-axis: log(thread_pool_size)   [T]
    #   y-axis: log(batch_size)           [B]
    #   z-axis: log(file_count)           [F]
    # and we color the points by log(time_in_seconds).
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df["log_thread_pool_size"],
        df["log_batch_size"],
        df["log_file_count"],
        c=df["logTotalTime"],
        cmap="viridis",
        s=50,
        alpha=0.8
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("log(time_in_seconds)")

    ax.set_xlabel("log(thread_pool_size)")
    ax.set_ylabel("log(batch_size)")
    ax.set_zlabel("log(file_count)")
    ax.set_title("3D Scatter Plot:\nLog(T), Log(B), Log(F) (Color = log(time_in_seconds))")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
