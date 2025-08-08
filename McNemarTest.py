import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

def create_predictions_from_confusion_matrix(cm):
    """Create prediction arrays from confusion matrix"""
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    y_true = np.array([0]*int(tn + fp) + [1]*int(fn + tp))
    y_pred = np.array([0]*int(tn) + [1]*int(fp) + [0]*int(fn) + [1]*int(tp))
    return y_true, y_pred

# Define confusion matrices for each method
methods = {
    'No_Sampling': {
        'XGBoost': np.array([[876, 131], [128, 569]]),
        'RandomForest': np.array([[887, 120], [170, 527]]),
        'Bagging': np.array([[891, 116], [157, 540]]),
        'GradientBoosting': np.array([[844, 163], [276, 421]])
    },
    'SMOTE': {
        'XGBoost': np.array([[829, 178], [139, 558]]),
        'RandomForest': np.array([[823, 184], [144, 553]]),
        'Bagging': np.array([[828, 179], [138, 559]]),
        'GradientBoosting': np.array([[730, 277], [173, 524]])
    },
    'BorderlineSMOTE': {
        'XGBoost': np.array([[914, 93], [158, 539]]),
        'RandomForest': np.array([[913, 94], [200, 497]]),
        'Bagging': np.array([[924, 83], [178, 519]]),
        'GradientBoosting': np.array([[834, 173], [254, 443]])
    },
    'ADASYN': {
        'XGBoost': np.array([[852, 155], [97, 600]]),
        'RandomForest': np.array([[842, 165], [119, 578]]),
        'Bagging': np.array([[851, 156], [101, 596]]),
        'GradientBoosting': np.array([[717, 290], [169, 528]])
    }
}

def calculate_additional_metrics(cm):
    """Calculate comprehensive metrics from confusion matrix"""
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total = np.sum(cm)
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    error_rate = (fp + fn) / total
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Error Rate': error_rate
    }

def create_comparison_visualization(results_df, output_path):
    """Create and save visualization of statistical comparison results"""
    # Set style for better visualization
    plt.style.use('default')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Heatmap of p-values - Handle duplicates by taking mean
    plt.subplot(2, 1, 1)
    pivot_df = results_df.pivot_table(
        index='Method 1', 
        columns='Method 2', 
        values='P-value',
        aggfunc='mean'  # Take mean if duplicates exist
    ).fillna(1.0)  # Fill NaN with 1.0 (no significance)
    
    # Create heatmap
    sns.heatmap(pivot_df, 
                annot=True, 
                cmap='RdYlGn_r', 
                vmin=0, 
                vmax=0.05, 
                center=0.025, 
                fmt='.4f',
                square=True)  # Make cells square
    plt.title('P-values Heatmap of Method Comparisons\n(Values < 0.05 indicate significant differences)',
             pad=20)
    
    # Bar plot of effect sizes - Group by methods and take mean
    plt.subplot(2, 1, 2)
    effect_sizes = results_df.groupby(['Method 1', 'Method 2'])['Effect Size'].mean().reset_index()
    sns.barplot(data=effect_sizes, 
                x='Method 1', 
                y='Effect Size',
                hue='Method 2', 
                palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Effect Sizes Between Methods', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure with high quality
    plt.savefig(f"{output_path}/statistical_comparison.png", 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    # Create additional visualizations for model-specific comparisons
    for model in results_df['Model'].unique():
        plt.figure(figsize=(10, 8))
        model_data = results_df[results_df['Model'] == model]
        
        pivot_df = model_data.pivot_table(
            index='Method 1',
            columns='Method 2',
            values='P-value',
            aggfunc='mean'
        ).fillna(1.0)
        
        sns.heatmap(pivot_df,
                   annot=True,
                   cmap='RdYlGn_r',
                   vmin=0,
                   vmax=0.05,
                   center=0.025,
                   fmt='.4f',
                   square=True)
        plt.title(f'P-values Heatmap for {model}\n(Values < 0.05 indicate significant differences)')
        plt.tight_layout()
        plt.savefig(f"{output_path}/statistical_comparison_{model}.png",
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        plt.close()# ... [Your existing code for confusion matrices and methods] ...

# Create output directory if it doesn't exist
output_path = Path("MoreData/Statistical_Tests")
output_path.mkdir(parents=True, exist_ok=True)

# Initialize results storage
results_list = []

def perform_mcnemar_test(pred1, pred2, y_true, method1_name, method2_name):
    """
    Perform McNemar's test between two classifiers
    
    Parameters:
    -----------
    pred1, pred2 : array-like
        Predictions from two different methods
    y_true : array-like
        True labels
    method1_name, method2_name : str
        Names of the methods being compared
        
    Returns:
    --------
    statistic : float
        McNemar's test statistic
    p_value : float
        p-value from the test
    """
    # Create contingency table
    correct1 = pred1 == y_true
    correct2 = pred2 == y_true
    
    # Calculate discordant predictions
    b = np.sum(~correct1 & correct2)  # method1 wrong, method2 right
    c = np.sum(correct1 & ~correct2)  # method1 right, method2 wrong
    
    # Create contingency table
    table = np.array([[0, b], [c, 0]])
    
    # Perform McNemar's test
    try:
        result = mcnemar(table, exact=True)
        return result.statistic, result.pvalue
    except ValueError as e:
        print(f"Error in McNemar test between {method1_name} and {method2_name}: {e}")
        return 0.0, 1.0
# Perform analysis and save results
with open(output_path / "detailed_comparison.txt", "w", encoding='utf-8') as f:
    f.write("Comprehensive Statistical Comparison of Oversampling Methods\n")
    f.write("====================================================\n\n")
    
    for model in ['XGBoost', 'RandomForest', 'Bagging', 'GradientBoosting']:
        f.write(f"\nResults for {model}\n")
        f.write("=" * 50 + "\n")
        
        # Define your desired order explicitly
        ordered_methods = ['No_Sampling', 'SMOTE', 'BorderlineSMOTE', 'ADASYN']
        
        for method1 in methods:
            for method2 in methods:
        # Iterate through pairs of methods
        #for i, method1 in enumerate(ordered_methods):
        #    for method2 in ordered_methods[i+1:]:
                if method1 < method2:
                    cm1 = methods[method1][model]
                    cm2 = methods[method2][model]
                    
                    # Calculate metrics
                    metrics1 = calculate_additional_metrics(cm1)
                    metrics2 = calculate_additional_metrics(cm2)
                    
                    # Perform McNemar's test
                    y_true1, y_pred1 = create_predictions_from_confusion_matrix(cm1)
                    y_true2, y_pred2 = create_predictions_from_confusion_matrix(cm2)
                    statistic, p_value = perform_mcnemar_test(y_pred1, y_pred2, y_true1, 
                                                            method1, method2)
                    
                    # Calculate effect size
                    effect_size = abs(metrics2['Accuracy'] - metrics1['Accuracy'])
                    
                    # Store results for visualization
                    results_list.append({
                        'Model': model,
                        'Method 1': method1,
                        'Method 2': method2,
                        'P-value': p_value,
                        'Effect Size': effect_size,
                        'Significant': p_value < 0.05
                    })
                    
                    # Write detailed results
                    f.write(f"\n{method1} vs {method2}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"McNemar's Test Statistic: {statistic:.4f}\n")
                    f.write(f"P-value: {p_value:.4f}\n")
                    f.write(f"Significant difference (alpha=0.05): {p_value < 0.05}\n")
                    f.write(f"Effect Size: {effect_size:.4f}\n\n")
                    
                    # Write performance metrics
                    f.write("Performance Metrics Comparison:\n")
                    for metric in metrics1.keys():
                        f.write(f"{metric:12} - {method1}: {metrics1[metric]:.4f}, "
                               f"{method2}: {metrics2[metric]:.4f}\n")
                    f.write("-" * 40 + "\n")

# Create DataFrame and save to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(output_path / "statistical_results.csv", index=False)

# Create visualizations
create_comparison_visualization(results_df, output_path)

print("Analysis completed. Results saved in MoreData/Statistical_Tests/")