import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
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
    # Plot 4: Linear model predicting TotalTime(ns) using our improved model,
    #         and comparing it to a simple model with only the log-transformed
    #         number of files feature.
    # =============================================================================
    
    # Drop rows with non-positive values for Concurrency, SimulatedCPUs, or TotalTime(ns)
    df = df[(df['Concurrency'] > 0) & (df['SimulatedCPUs'] > 0) & (df['TotalTime(ns)'] > 0)]
    
    # Create new features in log-space (adding 1 to NumFiles to handle zero values)
    df['log_NumFiles']    = np.log(df['NumFiles'] + 1)
    df['log_Concurrency']   = np.log(df['Concurrency'])
    df['log_SimulatedCPUs'] = np.log(df['SimulatedCPUs'])
    df['log_TotalTime']     = np.log(df['TotalTime(ns)'])
    
    # Create squared terms
    df['log_Concurrency_sq']   = df['log_Concurrency'] ** 2
    df['log_SimulatedCPUs_sq'] = df['log_SimulatedCPUs'] ** 2
    
    # Define predictors for the full model and the simple model
    features_full   = ['log_NumFiles', 'log_Concurrency', 'log_SimulatedCPUs',
                       'log_Concurrency_sq', 'log_SimulatedCPUs_sq']
    features_simple = ['log_NumFiles']
    
    # Define the response variable
    y = df['log_TotalTime']
    
    # For a consistent train-test split, use the DataFrame index
    train_idx, test_idx = train_test_split(df.index, test_size=0.3, random_state=42)
    
    # Create training and test sets for both models
    X_train_full   = df.loc[train_idx, features_full]
    X_test_full    = df.loc[test_idx,  features_full]
    X_train_simple = df.loc[train_idx, features_simple]
    X_test_simple  = df.loc[test_idx,  features_simple]
    y_train        = df.loc[train_idx, 'log_TotalTime']
    y_test         = df.loc[test_idx,  'log_TotalTime']
    
    # Add a constant term (intercept) to the predictors
    X_train_full_const   = sm.add_constant(X_train_full)
    X_test_full_const    = sm.add_constant(X_test_full)
    X_train_simple_const = sm.add_constant(X_train_simple)
    X_test_simple_const  = sm.add_constant(X_test_simple)
    
    # Fit the full model (with all predictors) and the simple model
    model_full   = sm.OLS(y_train, X_train_full_const).fit()
    model_simple = sm.OLS(y_train, X_train_simple_const).fit()
    
    # Print out the model summaries for comparison
    print("Full Model Summary:")
    print(model_full.summary())
    print("\nSimple Model Summary:")
    print(model_simple.summary())
    
    # Predict on the test sets for both models
    y_test_pred_full   = model_full.predict(X_test_full_const)
    y_test_pred_simple = model_simple.predict(X_test_simple_const)
    
    # Exponentiate the predictions and the actual values to convert back to the original scale
    y_test_pred_full_exp   = np.exp(y_test_pred_full)
    y_test_pred_simple_exp = np.exp(y_test_pred_simple)
    y_test_exp             = np.exp(y_test)
    
    # Compute Pearson correlation (and p-value) on the original scale for both models
    r_full,   p_full   = pearsonr(y_test_exp, y_test_pred_full_exp)
    r_simple, p_simple = pearsonr(y_test_exp, y_test_pred_simple_exp)
    
    # Plot actual vs. predicted TotalTime(ns) for both models
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Full model
    axs[0].scatter(y_test_exp, y_test_pred_full_exp, color='teal')
    axs[0].set_xlabel("Actual TotalTime(ns)")
    axs[0].set_ylabel("Predicted TotalTime(ns)")
    axs[0].set_title("Full Model Predictions (Test Set)")
    axs[0].text(0.05, 0.95, f"r = {r_full:.3f}\np = {p_full:.3e}\nR² = {model_full.rsquared:.3f}",
                transform=axs[0].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Right: Simple model (only log_NumFiles)
    axs[1].scatter(y_test_exp, y_test_pred_simple_exp, color='purple')
    axs[1].set_xlabel("Actual TotalTime(ns)")
    axs[1].set_ylabel("Predicted TotalTime(ns)")
    axs[1].set_title("Simple Model Predictions (Test Set)")
    axs[1].text(0.05, 0.95, f"r = {r_simple:.3f}\np = {p_simple:.3e}\nR² = {model_simple.rsquared:.3f}",
                transform=axs[1].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Plot raw coefficients for the full model (excluding the intercept if present)
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    # Check if 'const' exists in the parameters; if so, drop it.
    if 'const' in model_full.params.index:
        raw_betas_full = model_full.params.drop('const')
    else:
        raw_betas_full = model_full.params
    
    raw_betas_full.plot(kind='bar', color='skyblue', ax=ax2)
    ax2.set_ylabel("Beta Coefficient")
    ax2.set_title("Full Model Coefficients (Training Set)")
    plt.tight_layout()
    plt.show()


    # Define the candidate features to add to the baseline
    candidate_features = ['log_Concurrency', 'log_SimulatedCPUs', 'log_Concurrency_sq', 'log_SimulatedCPUs_sq']
    
    # Define the baseline predictor and response
    X_baseline = df[['log_NumFiles']]
    y = df['log_TotalTime']
    
    # Set the number of folds for cross-validation
    cv_folds = 5
    
    # Evaluate the baseline model using 5-fold CV
    baseline_cv_scores = cross_val_score(LinearRegression(), X_baseline, y, cv=cv_folds, scoring='r2')
    baseline_cv_mean = baseline_cv_scores.mean()
    baseline_cv_std = baseline_cv_scores.std()
    
    print("Baseline Model (using only 'log_NumFiles'):")
    print(f"  CV R²: {baseline_cv_mean:.4f} (± {baseline_cv_std:.4f})")
    print("-" * 50)
    
    # Loop over each candidate feature, add it to the baseline, and perform CV
    for feature in candidate_features:
        X_candidate = df[['log_NumFiles', feature]]
        cv_scores = cross_val_score(LinearRegression(), X_candidate, y, cv=cv_folds, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        improvement = cv_mean - baseline_cv_mean
        print(f"Model with features: log_NumFiles + {feature}")
        print(f"  CV R²: {cv_mean:.4f} (± {cv_std:.4f})  |  Improvement vs. baseline: {improvement:+.4f}")
        print("-" * 50)

if __name__ == '__main__':
    main()
