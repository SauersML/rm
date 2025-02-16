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
    df = pd.read_csv("test_results.csv", header=0, # Tell pandas to read header
                     names=["SimulatedCPUs", "NumFiles", "Concurrency", "TotalTime"])

    # Avoid issues with log(0)
    epsilon = 1e-6
    for col in ["SimulatedCPUs", "NumFiles", "Concurrency", "TotalTime"]:
        df[col] = df[col] + epsilon

    # Compute logged variables
    df["logSimulatedCPUs"] = np.log(df["SimulatedCPUs"])
    df["logNumFiles"] = np.log(df["NumFiles"])
    df["logConcurrency"] = np.log(df["Concurrency"])
    df["logTotalTime"] = np.log(df["TotalTime"])

    # =======================
    # 2. Feature Engineering for the Model
    # =======================
    # Create squared terms
    df["logSimulatedCPUs_sq"] = df["logSimulatedCPUs"] ** 2
    df["logNumFiles_sq"] = df["logNumFiles"] ** 2
    df["logConcurrency_sq"] = df["logConcurrency"] ** 2

    # Create interaction terms
    df["logSimulatedCPUs_x_logNumFiles"] = df["logSimulatedCPUs"] * df["logNumFiles"]
    df["logNumFiles_x_logConcurrency"] = df["logNumFiles"] * df["logConcurrency"]
    df["logSimulatedCPUs_x_logConcurrency"] = df["logSimulatedCPUs"] * df["logConcurrency"]
    df["logSimulatedCPUs_x_logNumFiles_x_logConcurrency"] = (
        df["logSimulatedCPUs"] * df["logNumFiles"] * df["logConcurrency"]
    )

    # Predictors list per the proposed model (without log(T)*log(B))
    predictors = [
        "logSimulatedCPUs",  # C
        "logNumFiles",  # F
        "logConcurrency",  # Con
        "logSimulatedCPUs_sq",  # C²
        "logNumFiles_sq",  # F²
        "logConcurrency_sq",  # Con²
        "logNumFiles_x_logConcurrency",   # F*Con
        "logSimulatedCPUs_x_logNumFiles_x_logConcurrency"  # C*F*Con
    ]
    # REMOVE logSimulatedCPUs_x_logNumFiles and logSimulatedCPUs_x_logConcurrency
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
    beta1 = ridge_ols_model.params["logSimulatedCPUs"]
    beta2 = ridge_ols_model.params["logNumFiles"]
    beta3 = ridge_ols_model.params["logConcurrency"]
    beta4 = ridge_ols_model.params["logSimulatedCPUs_sq"]
    beta5 = ridge_ols_model.params["logNumFiles_sq"]
    beta6 = ridge_ols_model.params["logConcurrency_sq"]
    beta7 = ridge_ols_model.params["logNumFiles_x_logConcurrency"]
    beta8 = ridge_ols_model.params["logSimulatedCPUs_x_logNumFiles_x_logConcurrency"]


    final_model_str = (
        "Final Chosen Ridge Model:\n" +
        "log(TotalTime) = {:.4f} + {:.4f}*log(C) + {:.4f}*log(F) + {:.4f}*log(Con) + \n".format(beta0, beta1, beta2, beta3) +
        "      {:.4f}*log(C)^2 + {:.4f}*log(F)^2 + {:.4f}*log(Con)^2 + \n".format(beta4, beta5, beta6) +
        "      {:.4f}*log(F)*log(Con) + {:.4f}*log(C)*log(F)*log(Con)".format(beta7, beta8)
    )

    print("\n" + final_model_str)

    # =======================
    # 5. 3D Visualization of the Data and Model Predictions
    # =======================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df["logSimulatedCPUs"],
        df["logNumFiles"],
        df["logConcurrency"],
        c=df["logTotalTime"],
        cmap="viridis",
        s=50,
        alpha=0.8
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("log(time_in_seconds)")

    ax.set_xlabel("log(SimulatedCPUs)")
    ax.set_ylabel("log(NumFiles)")
    ax.set_zlabel("log(Concurrency)")
    ax.set_title("3D Scatter Plot:\nLog(C), Log(F), Log(Con) (Color = log(TotalTime))")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
