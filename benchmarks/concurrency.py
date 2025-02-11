import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

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
    # Plot 4: Linear model predicting TotalTime(ns) using our improved model.
    #
    # The model is defined in log-space as:
    #   log(TotalTime(ns)) = β₀ + β₁*log(NumFiles) + β₂*log(Concurrency) + β₃*log(SimulatedCPUs)
    #                       + β₄*[log(Concurrency)]² + β₅*[log(SimulatedCPUs)]²
    #                       + β₆*(log(Concurrency)*log(SimulatedCPUs)) + ε
    #
    # We perform out-of-sample validation:
    #   - Split the data into training and test sets.
    #   - Fit the model on training data.
    #   - Predict on the test set and compute Pearson's r and p-value between actual and
    #     predicted TotalTime(ns) (after exponentiating back to the original scale).
    # Also, we plot:
    #   - A scatter plot of actual vs. predicted TotalTime(ns).
    #   - A bar plot of standardized beta coefficients.
    # =============================================================================
    # Create new features in log-space
    df['log_NumFiles'] = np.log(df['NumFiles'])
    df['log_Concurrency'] = np.log(df['Concurrency'])
    df['log_SimulatedCPUs'] = np.log(df['SimulatedCPUs'])
    df['log_Concurrency_sq'] = df['log_Concurrency']**2
    df['log_SimulatedCPUs_sq'] = df['log_SimulatedCPUs']**2
    df['log_TotalTime'] = np.log(df['TotalTime(ns)'])
    
    # Define predictors and response in log-space
    features = ['log_NumFiles', 'log_Concurrency', 'log_SimulatedCPUs',
                'log_Concurrency_sq', 'log_SimulatedCPUs_sq']
    X_new = df[features]
    y_new = df['log_TotalTime']
    
    # Split data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=42)
    
    # Add constant term to both training and testing predictors
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    # Fit the OLS model on the training data
    model = sm.OLS(y_train, X_train_const).fit()
    print(model.summary())
    
    # Predict on the test set
    y_test_pred = model.predict(X_test_const)
    
    # Exponentiate predictions and actual test responses to convert back to the original scale
    y_test_pred_exp = np.exp(y_test_pred)
    y_test_exp = np.exp(y_test)
    
    # Compute Pearson correlation and p-value on the original scale
    r, p = pearsonr(y_test_exp, y_test_pred_exp)
    
    # Compute standardized beta coefficients using training set statistics.
    # For each predictor (excluding the intercept), standardize by multiplying with (std(X) / std(y)).
    std_X_train = X_train.std()
    std_y_train = y_train.std()
    standardized_betas = model.params[1:] * (std_X_train / std_y_train)
    
    # Create a figure with two subplots:
    # Left: Scatter plot of actual vs. predicted TotalTime(ns) on the original scale.
    # Right: Bar plot of the standardized beta coefficients.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    axs[0].scatter(y_test_exp, y_test_pred_exp, color='teal')
    axs[0].set_xlabel("Actual TotalTime(ns)")
    axs[0].set_ylabel("Predicted TotalTime(ns)")
    axs[0].set_title("Actual vs Predicted TotalTime(ns) (Test Set)")
    # Annotate with Pearson correlation coefficient and p-value
    axs[0].text(0.05, 0.95, f"r = {r:.3f}\np = {p:.3e}",
                transform=axs[0].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    
    standardized_betas.plot(kind='bar', ax=axs[1], color='coral')
    axs[1].set_ylabel("Standardized Beta Coefficient")
    axs[1].set_title("Z-Normed Beta Coefficients (Training Set)")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
