import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap

def main():
    # Load data (no header in CSV)
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

    # For each file_count, select the row with the minimum time_in_seconds (optimal combo)
    optimal_idx = df.groupby("file_count")["time_in_seconds"].idxmin()
    optimal_mask = df.index.isin(optimal_idx)
    non_optimal_mask = ~optimal_mask

    # For non-optimal points, normalize colors based on log(time)
    non_optimal_time = df.loc[non_optimal_mask, "time_in_seconds"]
    log_non_optimal_time = np.log(non_optimal_time)
    norm = (log_non_optimal_time - log_non_optimal_time.min()) / (
        log_non_optimal_time.max() - log_non_optimal_time.min()
    )
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["orange", "yellow", "blue"])
    non_optimal_colors = custom_cmap(norm)

    # 3D Scatter Plot:
    # Axes: x = log(batch_size), y = log(thread_pool_size), z = log(file_count)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot non-optimal points (alpha to de-emphasize)
    ax.scatter(
        df.loc[non_optimal_mask, "log_batch_size"],
        df.loc[non_optimal_mask, "log_thread_pool_size"],
        df.loc[non_optimal_mask, "log_file_count"],
        c=non_optimal_colors,
        marker="o",
        alpha=0.1,
        label="Non-Optimal",
    )

    # Plot optimal points in red
    ax.scatter(
        df.loc[optimal_mask, "log_batch_size"],
        df.loc[optimal_mask, "log_thread_pool_size"],
        df.loc[optimal_mask, "log_file_count"],
        c="red",
        marker="o",
        label="Optimal (Lowest time)",
    )

    # ---- Regression on Optimal Points ----
    # Model: log(thread_pool_size) ~ log(batch_size) + log(file_count)
    optimal_data = df.loc[optimal_mask]
    X = optimal_data[["log_batch_size", "log_file_count"]]
    y = optimal_data["log_thread_pool_size"]
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    print(model.summary())

    # Draw a regression line: fix log(file_count) at its median, vary log(batch_size)
    fixed_log_file_count = optimal_data["log_file_count"].median()
    log_batch_range = np.linspace(
        optimal_data["log_batch_size"].min(), optimal_data["log_batch_size"].max(), 100
    )
    X_pred = pd.DataFrame({
        "const": 1,
        "log_batch_size": log_batch_range,
        "log_file_count": fixed_log_file_count,
    })
    predicted_log_thread_pool = model.predict(X_pred)

    # Plot the regression line (line in 3D at fixed log(file_count))
    ax.plot(
        log_batch_range,
        predicted_log_thread_pool,
        np.full_like(log_batch_range, fixed_log_file_count),
        color="green",
        linewidth=2,
        label="Best Fit: log(thread_pool_size)",
    )

    ax.set_xlabel("log(Batch Size)")
    ax.set_ylabel("log(Thread Pool Size)")
    ax.set_zlabel("log(File Count)")
    ax.set_title("3D Scatter: Log-Transformed Variables\nOptimal Points & Regression Line")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
