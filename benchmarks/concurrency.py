#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting even if not used directly
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr

def main():
    # =============================================================================
    # Load the data
    # =============================================================================
    df = pd.read_csv('test_results.csv')
    # The CSV is expected to have columns:
    # SimulatedCPUs, NumFiles, Concurrency, TotalTime(ns)
    
    # =============================================================================
    # Plot 1: 3D scatter plot (filtered)
    #
    # For each unique (SimulatedCPUs, NumFiles) pair, only keep the row having the
    # smallest TotalTime(ns). Then plot a 3D scatter with:
    #   x-axis: SimulatedCPUs
    #   y-axis: Concurrency
    #   z-axis: NumFiles
    # =============================================================================
    # Get the index for the minimum TotalTime(ns) for each (SimulatedCPUs, NumFiles) pair
    idx_min = df.groupby(["SimulatedCPUs", "NumFiles"])["TotalTime(ns)"].idxmin()
    df_min = df.loc[idx_min]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_min['SimulatedCPUs'], df_min['Concurrency'], df_min['NumFiles'],
               c='b', marker='o')
    ax.set_xlabel("SimulatedCPUs")
    ax.set_ylabel("Concurrency")
    ax.set_zlabel("NumFiles")
    ax.set_title("Plot 1: 3D Scatter of Filtered Data (Min TotalTime)")
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Plot 2: Scatter plot of Concurrency vs TotalTime(ns) with SimulatedCPUs as the color
    # =============================================================================
    plt.figure()
    scatter = plt.scatter(df['Concurrency'], df['TotalTime(ns)'], 
                          c=df['SimulatedCPUs'], cmap='viridis')
    plt.xlabel("Concurrency")
    plt.ylabel("TotalTime(ns)")
    plt.title("Plot 2: Concurrency vs TotalTime(ns) (Color: SimulatedCPUs)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("SimulatedCPUs")
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Plot 3: Scatter plot of Concurrency vs TotalTime(ns) with NumFiles as the color
    # =============================================================================
    plt.figure()
    scatter = plt.scatter(df['Concurrency'], df['TotalTime(ns)'], 
                          c=df['NumFiles'], cmap='plasma')
    plt.xlabel("Concurrency")
    plt.ylabel("TotalTime(ns)")
    plt.title("Plot 3: Concurrency vs TotalTime(ns) (Color: NumFiles)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("NumFiles")
    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Plot 4: Linear model predicting TotalTime(ns) from SimulatedCPUs, NumFiles, and Concurrency.
    #          Also plot:
    #             - The scatter plot of actual vs predicted TotalTime(ns) with correlation (r) and p-value.
    #             - The standardized (z-normed) beta coefficients for each predictor.
    # =============================================================================
    # Prepare the data for regression:
    X = df[['SimulatedCPUs', 'NumFiles', 'Concurrency']]
    y = df['TotalTime(ns)']
    X_with_const = sm.add_constant(X)  # adds the intercept term

    # Fit the linear regression model
    model = sm.OLS(y, X_with_const).fit()
    print(model.summary())  # print model summary to the console

    # Compute predictions
    y_pred = model.predict(X_with_const)
    
    # Compute Pearson correlation (r) and corresponding p-value between actual and predicted values
    r, p = pearsonr(y, y_pred)

    # Calculate standardized beta coefficients.
    # The formula: beta_standardized = beta_unstandardized * (std(X) / std(y))
    std_X = X.std()
    std_y = y.std()
    # Exclude the intercept (constant) term when standardizing
    standardized_betas = model.params[1:] * (std_X / std_y)

    # Create a figure with two subplots:
    # Left: Scatter plot of actual vs. predicted TotalTime(ns)
    # Right: Bar plot of the standardized beta coefficients.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Actual vs Predicted scatter plot
    axs[0].scatter(y, y_pred, color='teal')
    axs[0].set_xlabel("Actual TotalTime(ns)")
    axs[0].set_ylabel("Predicted TotalTime(ns)")
    axs[0].set_title("Actual vs Predicted TotalTime(ns)")
    # Annotate with correlation coefficient and p-value
    axs[0].text(0.05, 0.95, f"r = {r:.3f}\np = {p:.3e}",
                transform=axs[0].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Right subplot: Bar plot for standardized (z-normed) beta coefficients
    standardized_betas.plot(kind='bar', ax=axs[1], color='coral')
    axs[1].set_ylabel("Standardized Beta Coefficient")
    axs[1].set_title("Z-Normed Beta Coefficients")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
