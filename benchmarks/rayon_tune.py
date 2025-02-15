import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import cvxpy as cp

def main():
    # === Load and Prepare Data ===
    # Load CSV data (no header) and assign column names
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

    # === Build a Predictive Quadratic Model for log_time ===
    # Include quadratic terms and interaction terms between the log variables.
    df["log_thread_pool_size_sq"] = df["log_thread_pool_size"] ** 2
    df["log_batch_size_sq"] = df["log_batch_size"] ** 2
    df["log_thread_pool_batch"] = df["log_thread_pool_size"] * df["log_batch_size"]

    # Include interactions between log_file_count and the other two variables,
    # so that the optimal settings can vary with file count.
    df["log_file_thread_pool"] = df["log_file_count"] * df["log_thread_pool_size"]
    df["log_file_batch"] = df["log_file_count"] * df["log_batch_size"]

    # Define the regression features and add a constant
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

    # Fit the quadratic regression model
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # === Extract Model Coefficients ===
    # Our model is:
    # log_time = beta0 + beta1*log_file_count
    #          + beta2*log_thread_pool_size + beta3*log_batch_size
    #          + beta4*(log_thread_pool_size)^2 + beta5*(log_batch_size)^2
    #          + beta6*(log_thread_pool_size*log_batch_size)
    #          + beta7*(log_file_count*log_thread_pool_size)
    #          + beta8*(log_file_count*log_batch_size)
    beta0 = model.params["const"]
    beta1 = model.params["log_file_count"]
    beta2 = model.params["log_thread_pool_size"]
    beta3 = model.params["log_batch_size"]
    beta4 = model.params["log_thread_pool_size_sq"]
    beta5 = model.params["log_batch_size_sq"]
    beta6 = model.params["log_thread_pool_batch"]
    beta7 = model.params["log_file_thread_pool"]
    beta8 = model.params["log_file_batch"]

    # === Set Up the Convex Optimization Problem ===
    # For a fixed log_file_count (denoted LFC), the model becomes:
    # log_time = (beta0 + beta1*LFC) +
    #            (beta2 + beta7*LFC)*x + (beta3 + beta8*LFC)*y +
    #            beta4*x^2 + beta5*y^2 + beta6*x*y,
    # where x = log_thread_pool_size and y = log_batch_size.
    # Since the constant term doesn't affect the minimization, the effective
    # objective (in x and y) is:
    #   f(x,y) = (beta2 + beta7*LFC)*x + (beta3 + beta8*LFC)*y + beta4*x^2 + beta5*y^2 + beta6*x*y.
    #
    # Thread_pool_size and batch_size are ≥1
    def optimize_for_log_file_count(LFC):
        x = cp.Variable()  # log_thread_pool_size
        y = cp.Variable()  # log_batch_size
        # Define the objective function (ignoring constant terms independent of x,y)
        obj = ((beta2 + beta7 * LFC) * x +
               (beta3 + beta8 * LFC) * y +
               beta4 * cp.square(x) +
               beta5 * cp.square(y) +
               beta6 * x * y)
        objective = cp.Minimize(obj)
        constraints = [x >= 0, y >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return x.value, y.value

    # === Solve for the Optimal Configuration for a Range of log_file_count Values ===
    LFC_values = np.linspace(df["log_file_count"].min(), df["log_file_count"].max(), 50)
    optimal_log_thread_pool = []  # will store optimal x values
    optimal_log_batch = []         # will store optimal y values
    for LFC in LFC_values:
        opt_x, opt_y = optimize_for_log_file_count(LFC)
        optimal_log_thread_pool.append(opt_x)
        optimal_log_batch.append(opt_y)

    # === Plotting ===
    # We'll create a 3D scatter plot of the original data points along with
    # the optimal configuration curve (in red) as a function of log_file_count.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot original data points (using log_batch_size as x, log_thread_pool_size as y,
    # and log_file_count as z)
    ax.scatter(
        df["log_batch_size"],
        df["log_thread_pool_size"],
        df["log_file_count"],
        c="gray",
        marker="o",
        alpha=0.3,
        label="Data Points",
    )

    # Plot the optimal curve. Our optimizer returns:
    #   x = log_thread_pool_size, y = log_batch_size.
    # When plotting, we use:
    #   x-axis: log(batch_size)  (optimal_log_batch)
    #   y-axis: log(thread_pool_size) (optimal_log_thread_pool)
    #   z-axis: log(file_count) (LFC_values)
    ax.plot(
        optimal_log_batch,
        optimal_log_thread_pool,
        LFC_values,
        color="red",
        linewidth=3,
        label="Optimal Configuration",
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
