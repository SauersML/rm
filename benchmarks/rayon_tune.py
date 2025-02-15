import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import cvxpy as cp
from matplotlib.colors import LinearSegmentedColormap

def main():
    # === 1. Load and Prepare Data ===
    df = pd.read_csv(
        "rayon_deletion_benchmark.csv",
        header=None,
        names=["file_count", "time_in_seconds", "thread_pool_size", "batch_size"],
    )

    # Create log-transformed variables
    df["log_file_count"] = np.log(df["file_count"])
    df["log_time"] = np.log(df["time_in_seconds"])
    df["log_thread_pool_size"] = np.log(df["thread_pool_size"])
    df["log_batch_size"] = np.log(df["batch_size"])

    # === 2. Build a Quadratic Regression Model for log_time ===
    # Add quadratic and interaction terms
    df["log_thread_pool_size_sq"] = df["log_thread_pool_size"] ** 2
    df["log_batch_size_sq"] = df["log_batch_size"] ** 2
    df["log_thread_pool_batch"] = df["log_thread_pool_size"] * df["log_batch_size"]
    df["log_file_thread_pool"] = df["log_file_count"] * df["log_thread_pool_size"]
    df["log_file_batch"] = df["log_file_count"] * df["log_batch_size"]

    # Define features and add constant
    X = df[
        [
            "log_file_count",
            "log_thread_pool_size",
            "log_batch_size",
            "log_thread_pool_size_sq",
            "log_batch_size_sq",
            "log_thread_pool_batch",
            "log_file_thread_pool",
            "log_file_batch",
        ]
    ]
    X = sm.add_constant(X)
    y = df["log_time"]

    # Fit the regression model (convex least-squares)
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # === 3. Extract Coefficients ===
    # Fitted model:
    # log_time = beta0 + beta1*log_file_count +
    #            beta2*log_thread_pool_size + beta3*log_batch_size +
    #            beta4*(log_thread_pool_size)^2 + beta5*(log_batch_size)^2 +
    #            beta6*(log_thread_pool_size * log_batch_size) +
    #            beta7*(log_file_count * log_thread_pool_size) +
    #            beta8*(log_file_count * log_batch_size)
    beta0 = model.params["const"]
    beta1 = model.params["log_file_count"]
    beta2 = model.params["log_thread_pool_size"]
    beta3 = model.params["log_batch_size"]
    beta4 = model.params["log_thread_pool_size_sq"]
    beta5 = model.params["log_batch_size_sq"]
    beta6 = model.params["log_thread_pool_batch"]
    beta7 = model.params["log_file_thread_pool"]
    beta8 = model.params["log_file_batch"]

    # === 4. Set Up the Convex Optimization Problem ===
    # For a fixed log_file_count (LFC), we optimize over:
    #   x = log_thread_pool_size and y = log_batch_size.
    # Ignoring constant terms, the objective is:
    # f(x, y) = (beta2 + beta7*LFC)*x + (beta3 + beta8*LFC)*y +
    #           beta4*x^2 + beta5*y^2 + beta6*x*y
    # We express the quadratic part as a quadratic form:
    # Let Q = [[beta4, beta6/2], [beta6/2, beta5]]
    Q = np.array([[beta4, beta6 / 2.0],
                  [beta6 / 2.0, beta5]])

    def optimize_for_log_file_count(LFC):
        # Define variables: x = log_thread_pool_size, y = log_batch_size
        x = cp.Variable()
        y = cp.Variable()
        # Linear component
        lin_term = (beta2 + beta7 * LFC) * x + (beta3 + beta8 * LFC) * y
        # Quadratic component via quadratic form
        z = cp.vstack([x, y])
        quad_term = cp.quad_form(z, Q)
        # Objective: minimize the sum of the linear and quadratic terms
        objective = cp.Minimize(lin_term + quad_term)
        # Constraints: x, y >= 0 (to ensure original values are ≥ 1)
        constraints = [x >= 0, y >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return x.value, y.value

    # === 5. Solve for the Optimal Configuration over a Range of log_file_count ===
    LFC_values = np.linspace(df["log_file_count"].min(), df["log_file_count"].max(), 50)
    optimal_log_thread_pool = []
    optimal_log_batch = []
    for LFC in LFC_values:
        opt_x, opt_y = optimize_for_log_file_count(LFC)
        optimal_log_thread_pool.append(opt_x)
        optimal_log_batch.append(opt_y)

    # === 6. Plot the Results ===
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Color data points based on log_time using a custom colormap
    norm = (df["log_time"] - df["log_time"].min()) / (df["log_time"].max() - df["log_time"].min())
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["orange", "yellow", "blue"])
    colors = custom_cmap(norm)

    # Plot original data points:
    # X-axis: log(Batch Size), Y-axis: log(Thread Pool Size), Z-axis: log(File Count)
    sc = ax.scatter(
        df["log_batch_size"],
        df["log_thread_pool_size"],
        df["log_file_count"],
        c=colors,
        marker="o",
        alpha=0.7,
        label="Data Points"
    )

    # Add a colorbar to indicate log(time)
    mappable = plt.cm.ScalarMappable(cmap=custom_cmap)
    mappable.set_array(df["log_time"])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label("log(Time)")

    # Plot the optimal configuration curve (red line)
    ax.plot(
        optimal_log_batch,
        optimal_log_thread_pool,
        LFC_values,
        color="red",
        linewidth=3,
        label="Optimal Configuration"
    )

    ax.set_xlabel("log(Batch Size)")
    ax.set_ylabel("log(Thread Pool Size)")
    ax.set_zlabel("log(File Count)")
    ax.set_title("Optimal Configuration: Minimizing log(Time)")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
