import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

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


def create_predictions_from_confusion_matrix(cm):
    """Create prediction arrays from confusion matrix"""
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    y_true = np.array([0]*int(tn + fp) + [1]*int(fn + tp))
    y_pred = np.array([0]*int(tn) + [1]*int(fp) + [0]*int(fn) + [1]*int(tp))
    return y_true, y_pred

# Define confusion matrices and F1-scores for each method
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

# F1-scores from your figure data
f1_scores = {
    'XGBoost': {
        'No_Sampling': 0.8146,
        'SMOTE': 0.7788,
        'BorderlineSMOTE': 0.8111,
        'ADASYN': 0.8264
    },
    'RandomForest': {
        'No_Sampling': 0.7842,
        'SMOTE': 0.7713,
        'BorderlineSMOTE': 0.7717,
        'ADASYN': 0.8028
    },
    'Bagging': {
        'No_Sampling': 0.7982,
        'SMOTE': 0.7791,
        'BorderlineSMOTE': 0.7991,
        'ADASYN': 0.8226
    },
    'GradientBoosting': {
        'No_Sampling': 0.6573,
        'SMOTE': 0.6996,
        'BorderlineSMOTE': 0.6748,
        'ADASYN': 0.6970
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

def cohen_h_magnitude(h):
    """Classify effect size magnitude using Cohen's h thresholds"""
    if abs(h) < 0.01: return "Negligible"
    elif abs(h) < 0.2: return "Small"
    elif abs(h) < 0.5: return "Medium"
    else: return "Large"

def get_performance_direction(row, f1_scores):
    """Determine which method performed better based on F1-score"""
    if not row['Significant']:
        return "Equivalent"
    
    f1_1 = f1_scores[row['Model']][row['Method 1']]
    f1_2 = f1_scores[row['Model']][row['Method 2']]
    
    if f1_1 > f1_2 + 0.01:  # Adding small threshold for practical significance
        return f"{row['Method 1']} Better"
    elif f1_2 > f1_1 + 0.01:
        return f"{row['Method 2']} Better"
    else:
        return "Equivalent (Practically)"

def create_enhanced_comparison_visualization(results_df, output_path):
    """Create enhanced visualizations with magnitude and performance info"""
    #plt.style.use('seaborn')
    plt.style.use('seaborn-v0_8')
    
    # Create scatter plot of effect sizes vs p-values
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=results_df,
        x='Effect Size',
        y='P-value',
        hue='Magnitude',
        style='Performance',
        s=100,
        palette='viridis'
    )
    
    # Add significance thresholds
    plt.axhline(0.05, color='r', linestyle='--', label='Significance Threshold (α=0.05)')
    plt.axvline(0.01, color='g', linestyle=':', label='Small Effect Threshold (h=0.01)')
    
    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Practical Significance of Method Comparisons\n(Effect Size vs P-value)', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_path}/practical_significance.png", 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    # Create F1-score comparison heatmap
    f1_data = []
    for model in results_df['Model'].unique():
        for method in methods:
            f1_data.append({
                'Model': model,
                'Method': method,
                'F1-Score': f1_scores[model][method]
            })
    
    f1_df = pd.DataFrame(f1_data)
    plt.figure(figsize=(10, 6))
    pivot_f1 = f1_df.pivot(index='Model', columns='Method', values='F1-Score')
    sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': 'F1-Score'})
    plt.title('F1-Score Comparison Across Models and Methods', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_path}/f1_comparison.png", dpi=300)
    plt.close()

# Create output directory
output_path = Path("MoreData/Enhanced_Statistical_Tests")
output_path.mkdir(parents=True, exist_ok=True)

# Initialize results storage
enhanced_results = []

# Perform analysis
for model in ['XGBoost', 'RandomForest', 'Bagging', 'GradientBoosting']:
    for method1 in methods:
        for method2 in methods:
            if method1 < method2:
                cm1 = methods[method1][model]
                cm2 = methods[method2][model]
                
                # Calculate metrics
                metrics1 = calculate_additional_metrics(cm1)
                metrics2 = calculate_additional_metrics(cm2)
                
                # Perform McNemar's test
                y_true1, y_pred1 = create_predictions_from_confusion_matrix(cm1)
                y_true2, y_pred2 = create_predictions_from_confusion_matrix(cm2)
                statistic, p_value = perform_mcnemar_test(y_pred1, y_pred2, y_true1, method1, method2)
                
                # Calculate effect size
                effect_size = metrics2['Accuracy'] - metrics1['Accuracy']  # Signed effect size
                abs_effect_size = abs(effect_size)
                
                # Determine magnitude and performance
                magnitude = cohen_h_magnitude(abs_effect_size)
                performance = get_performance_direction({
                    'Model': model,
                    'Method 1': method1,
                    'Method 2': method2,
                    'Significant': p_value < 0.05,
                    'Effect Size': effect_size
                }, f1_scores)
                
                # Store enhanced results
                enhanced_results.append({
                    'Model': model,
                    'Method 1': method1,
                    'Method 2': method2,
                    'P-value': p_value,
                    'Effect Size': effect_size,
                    'Absolute Effect Size': abs_effect_size,
                    'Significant': p_value < 0.05,
                    'Magnitude': magnitude,
                    'Performance': performance,
                    'F1 Method1': f1_scores[model][method1],
                    'F1 Method2': f1_scores[model][method2],
                    'F1 Difference': f1_scores[model][method2] - f1_scores[model][method1]
                })

# Create enhanced DataFrame
enhanced_df = pd.DataFrame(enhanced_results)

# Save results
enhanced_df.to_csv(output_path / "enhanced_statistical_results.csv", index=False)

# Create enhanced visualizations
create_enhanced_comparison_visualization(enhanced_df, output_path)

# Generate summary report
with open(output_path / "summary_report.txt", "w") as f:
    f.write("Key Findings:\n")
    f.write("1. ADASYN showed consistent F1-score improvements:\n")
    adasyn_results = enhanced_df[
        (enhanced_df['Method 2'] == 'ADASYN') & 
        (enhanced_df['Performance'].str.contains('ADASYN Better'))
    ]
    for _, row in adasyn_results.iterrows():
        f.write(f"   - {row['Model']}: +{row['F1 Difference']:.4f} over {row['Method 1']} (p={row['P-value']:.2e}, h={row['Absolute Effect Size']:.4f})\n")
    
    f.write("\n2. BorderlineSMOTE showed specific advantages:\n")
    borderline_results = enhanced_df[
        (enhanced_df['Method 2'] == 'BorderlineSMOTE') & 
        (enhanced_df['Performance'].str.contains('BorderlineSMOTE Better'))
    ]
    for _, row in borderline_results.iterrows():
        f.write(f"   - {row['Model']}: +{row['F1 Difference']:.4f} over {row['Method 1']} (p={row['P-value']:.4f}, h={row['Absolute Effect Size']:.4f})\n")
    
    f.write("\n3. SMOTE generally underperformed:\n")
    smote_results = enhanced_df[
        (enhanced_df['Method 1'] == 'SMOTE') & 
        (enhanced_df['Performance'].str.contains('Better'))
    ]
    for _, row in smote_results.iterrows():
        f.write(f"   - {row['Method 2']} outperformed SMOTE in {row['Model']} by +{abs(row['F1 Difference']):.4f}\n")

print("Enhanced analysis completed. Results saved in MoreData/Enhanced_Statistical_Tests/")