## This code does not have additional machine learning techniques with comments.
## I have removed all the parts where feature importance was found using techniques other
## shap.
## This part has the weighted sum is given.

## In this part, I will use the weighted sum ranking to find the accuracy of 
## Top1, top2, top3, top4, top5 features and show them in a table.
## I also removed the part of borda count for now. It is present in new4.py.

## I have added the part for error analysis here after using weighted sum ranking.
## New:: Instead of using Dayoftheyear, I am using day of the season instead.
## New:: Dropping Month also from the dataset to not cause any confusion.

from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import numpy as np
import pandas as pd
import os
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use a non-interactive backend
from xgboost import XGBClassifier
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import seaborn as sns
"""
# Dictionary for drought-affected years (1 for drought, 0 for no drought).
# Original Analysis: Using the news reports from different newspapers.(84 Percent) 
drought_labels = {
    'Jodhpur': {2016: 1, 2017: 0, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Amravati': {2016: 1, 2017: 0, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Thanjavur': {2016: 1, 2017: 1, 2018: 0, 2019: 1, 2020: 0, 2021: 0}
}

# Dictionary for drought-affected years (1 for drought, 0 for no drought).
# Using the IMD reports with using SPI average from (Oct - May) 
drought_labels = {
    'Jodhpur': {2016: 1, 2017: 0, 2018: 1, 2019: 0, 2020: 0, 2021: 0},
    'Amravati': {2016: 1, 2017: 1, 2018: 0, 2019: 1, 2020: 0, 2021: 1},
    'Thanjavur': {2016: 1, 2017: 0, 2018: 1, 2019: 1, 2020: 0, 2021: 0}
}

# Last Try to do it again(86%)
drought_labels = {
    'Jodhpur': {2016: 1, 2017: 0, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Amravati': {2016: 1, 2017: 0, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Thanjavur': {2016: 0, 2017: 1, 2018: 0, 2019: 1, 2020: 0, 2021: 0}
}

#(90%)
drought_labels = {
    'Jodhpur': {2016: 1, 2017: 0, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Amravati': {2016: 1, 2017: 1, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Thanjavur': {2016: 0, 2017: 1, 2018: 0, 2019: 1, 2020: 0, 2021: 0}
}


# Dictionary for drought-affected years (1 for drought, 0 for no drought).
# Using newspaper and other government reports with Jit.
drought_labels = {
    'Jodhpur': {2016: 1, 2017: 0, 2018: 0, 2019: 1, 2020: 1, 2021:0 },
    'Amravati': {2016: 1, 2017: 0, 2018: 0, 2019: 1, 2020: 0, 2021:0 },
    'Thanjavur': {2016: 0, 2017: 1, 2018: 0, 2019: 1, 2020: 0, 2021:0 }
}
# Last Try to do it again(87%)
drought_labels = {
    'Jodhpur': {2016: 1, 2017: 0, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Amravati': {2016: 1, 2017: 1, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Thanjavur': {2016: 0, 2017: 1, 2018: 0, 2019: 1, 2020: 0, 2021: 0}
}

"""
performance_output_file = "MoreData/ResultsWeightedSum/performance.txt"

# Dictionary for drought-affected years (1 for drought, 0 for no drought).
# Using newspaper and other government reports with Jit.
drought_labels = {
    'Jodhpur': {2016: 1, 2017: 0, 2018: 0, 2019: 1, 2020: 1, 2021:0, 2022:1, 2023:0, 2024:1, 2025:0 },
    'Amravati': {2016: 1, 2017: 0, 2018: 0, 2019: 1, 2020: 0, 2021:0, 2022:0, 2023:1, 2024:1, 2025:0 },
    'Thanjavur': {2016: 0, 2017: 1, 2018: 0, 2019: 1, 2020: 0, 2021:0, 2022:0, 2023:0, 2024:1, 2025:0 }
}

def load_data(district):
    dfs = []
    for year in range(2016, 2026):  # Loop through the relevant years
        # Define filenames for both current and previous year's data
        current_file = f"MoreData/timeSeriesData/TimeSeries_{district}_{year}.csv"
        previous_file = f"MoreData/timeSeriesData/TimeSeries_{district}_{year-1}.csv"
        
        # Initialize an empty DataFrame for the current season
        season_df = pd.DataFrame()
        
        # Load data from October-December of the previous year
        if os.path.exists(previous_file):
            prev_df = pd.read_csv(previous_file)
            prev_df['date'] = pd.to_datetime(prev_df['date'])
            prev_df = prev_df[(prev_df['date'] >= f"{year-1}-10-01") & (prev_df['date'] <= f"{year-1}-12-31")]
            season_df = pd.concat([season_df, prev_df], ignore_index=True)
        
        # Load data from January-April of the current year
        if os.path.exists(current_file):
            curr_df = pd.read_csv(current_file)
            curr_df['date'] = pd.to_datetime(curr_df['date'])
            curr_df = curr_df[(curr_df['date'] >= f"{year}-01-01") & (curr_df['date'] <= f"{year}-04-30")]
            season_df = pd.concat([season_df, curr_df], ignore_index=True)
        
        # If we have data for the current season
        if not season_df.empty:
            # Assign the Rabi season year
            season_df['SeasonYear'] = year
            
            # Calculate the day of the season (1 to 213)
            start_date = pd.Timestamp(f"{year-1}-10-01")
            season_df['DayOfSeason'] = (season_df['date'] - start_date).dt.days + 1
            
            # Extract additional features
            season_df['Year'] = season_df['date'].dt.year
            season_df['Month'] = season_df['date'].dt.month
            
            # Drop unnecessary columns
            if 'system:index' in season_df.columns:
                season_df = season_df.drop(columns=['system:index'])
            season_df = season_df.drop(columns=['date', '.geo'], errors='ignore')
            
            # Add district column
            season_df['District'] = district
            
            # Add drought label for the Rabi season
            season_df['Drought'] = drought_labels[district][year]
            
            dfs.append(season_df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()



# Load the data for all districts
districts = ['Jodhpur', 'Amravati', 'Thanjavur']
data = pd.DataFrame()

# Process data for each district
for district in districts:
    district_data = load_data(district)
    data = pd.concat([data, district_data], ignore_index=True)

# Vectors to store metrics
model_names = []
accuracies = []
precisions = []
recalls = []

# Save to CSV
file_path = 'MoreData/ResultsWeightedSum/output_file.csv'
data.to_csv(file_path, index=False)

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define features (remote sensing indices) and target (drought affected or not)
X = data

#X = data.drop(columns=['SeasonYear','Drought', 'Year', 'Month', 'District'])  # Features
y = data['Drought']  # Target

# Save the column names before scaling
column_names = X.columns

# Define the columns to be normalized (first 12 features)
features_to_normalize = X.columns[:12]

# Normalize the first 12 features. 
# The rest of the features remain unchanged
scaler = MinMaxScaler()
X[features_to_normalize] = scaler.fit_transform(X[features_to_normalize])

# Convert back to a pandas DataFrame with the correct column names
X = pd.DataFrame(X, columns=column_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

X_train = X_train.drop(columns=['SeasonYear','Drought', 'Year', 'Month', 'District'])  # Features
X_test = X_test.drop(columns=['SeasonYear','Drought', 'Year', 'Month', 'District'])  # Features

# Handle class imbalance with SMOTEENN
# Note: SMOTEENN is working better than below commented BorderlineSMOTE.
#smote_enn = SMOTEENN(random_state=42)
#X_train, y_train = smote_enn.fit_resample(X_train, y_train)

# Define the BorderlineSMOTE resampling and ENN cleaning
#borderline_smote = BorderlineSMOTE(random_state=42)
#enn = EditedNearestNeighbours()

# Create a pipeline to apply both BorderlineSMOTE and ENN
#pipeline = Pipeline([('borderline_smote', borderline_smote), ('enn', enn)])

# Handle class imbalance with Borderline-SMOTE and ENN
#X_train, y_train = pipeline.fit_resample(X_train, y_train)

# Check class distribution before ADASYN
#print("\nOriginal class distribution:")
#print(y_train.value_counts())

# Handle class imbalance with ADASYN
#adasyn = ADASYN(random_state=42)
#X_train, y_train = adasyn.fit_resample(X_train, y_train)

## *****************************************************************************************************************##

### XGBoost Classifier ###
model_names.append('XGBoost')
xgb_model = XGBClassifier(n_estimators=200, random_state=42)

# Train-Test Split Results
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_xgb)
test_precision = precision_score(y_test, y_pred_xgb)
test_recall = recall_score(y_test, y_pred_xgb)

# Cross-validation Results
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = cross_val_score(XGBClassifier(n_estimators=200, random_state=42), X_train, y_train, cv=cv, scoring='accuracy')
cv_precisions = cross_val_score(XGBClassifier(n_estimators=200, random_state=42), X_train, y_train, cv=cv, scoring='precision')
cv_recalls = cross_val_score(XGBClassifier(n_estimators=200, random_state=42), X_train, y_train, cv=cv, scoring='recall')

# Calculate means for storing
accuracy_xgb = cv_accuracies.mean()
precision_xgb = cv_precisions.mean()
recall_xgb = cv_recalls.mean()

accuracies.append(test_accuracy)
precisions.append(test_precision)
recalls.append(test_recall)

# Print and save results
print("\nXGBoost Results")
print("================")
print("Cross-validation Results:")
print(f"CV Accuracy: {accuracy_xgb:.4f} ± {cv_accuracies.std():.4f}")
print(f"CV Precision: {precision_xgb:.4f} ± {cv_precisions.std():.4f}")
print(f"CV Recall: {recall_xgb:.4f} ± {cv_recalls.std():.4f}")
print("\nTrain-Test Split Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Save results to file
with open(performance_output_file, "w") as f:
    f.write("XGBoost Results\n")
    f.write("================\n\n")
    f.write("Cross-validation Results:\n")
    f.write(f"CV Accuracy: {accuracy_xgb:.4f} ± {cv_accuracies.std():.4f}\n")
    f.write(f"CV Precision: {precision_xgb:.4f} ± {cv_precisions.std():.4f}\n")
    f.write(f"CV Recall: {recall_xgb:.4f} ± {cv_recalls.std():.4f}\n\n")
    f.write("Train-Test Split Results:\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n\n")

# Create and save a summary table
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Cross-validation': [
        f"{accuracy_xgb:.4f} ± {cv_accuracies.std():.4f}",
        f"{precision_xgb:.4f} ± {cv_precisions.std():.4f}",
        f"{recall_xgb:.4f} ± {cv_recalls.std():.4f}"
    ],
    'Train-Test Split': [
        f"{test_accuracy:.4f}",
        f"{test_precision:.4f}",
        f"{test_recall:.4f}"
    ]
})

# Save metrics table as image
plt.figure(figsize=(10, 4))
ax = plt.gca()
ax.axis('tight')
ax.axis('off')
table = plt.table(cellText=metrics_df.values,
                 colLabels=metrics_df.columns,
                 rowLabels=None,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title("XGBoost Performance Metrics Comparison")
plt.savefig("MoreData/ResultsWeightedSum/xgboost_metrics_comparison.png", 
            bbox_inches='tight', dpi=300)
plt.close()

# Convert predictions to a DataFrame
results_df = X_test_copy
results_df["Actual_Label"] = y_test
results_df["Predicted_Label"] = y_pred_xgb

# Group by district and year
grouped_results = results_df.groupby(["District", "SeasonYear"])

# Prepare the output file
output_file = "MoreData/ResultsWeightedSum/xgboost_test_results.txt"
correct_groups = 0
total_groups = len(grouped_results)

with open(output_file, "w") as f:
    for (district, seasonYear), group in grouped_results:
        f.write(f"District: {district}, SeasonYear: {seasonYear} \n")
        f.write(f"Total Test Cases: {len(group)}\n")
       
        # Perform voting on the predicted labels
        predicted_counts = group['Predicted_Label'].value_counts()
        count_0 = predicted_counts.get(0, 0)
        count_1 = predicted_counts.get(1, 0)

         # Determine the majority prediction
        majority_prediction = 1 if count_1 >= count_0 else 0

        # Check if the majority prediction matches the actual label
        actual_label = group['Actual_Label'].mode()[0]
        if majority_prediction == actual_label:
            f.write("Group Detected Correctly\n")
            correct_groups += 1
        else:
            f.write("Group Detected Wrongly\n")

        # Write the voting results
        f.write(f"Predicted 0: {count_0}, Predicted 1: {count_1}\n")

        f.write("Actual vs Predicted:\n")
        for idx, row in group.iterrows():
            f.write(f"Actual: {row['Actual_Label']}, Predicted: {row['Predicted_Label']}\n")
        f.write("\n")

# Calculate accuracy, precision, and recall for the groups
accuracy = correct_groups / total_groups
precision = correct_groups / (correct_groups + (total_groups - correct_groups))
recall = correct_groups / total_groups

# Save the overall results to the file
with open(output_file, "w") as f:
    f.write("Overall Group Detection Results\n")
    f.write("===============================\n")
    f.write(f"Total Groups: {total_groups}\n")
    f.write(f"Correctly Detected Groups: {correct_groups}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")

with open(performance_output_file, "a") as f:
    f.write("Overall Group Detection Results\n")
    f.write("===============================\n")
    f.write(f"Total Groups: {total_groups}\n")
    f.write(f"Correctly Detected Groups: {correct_groups}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n\n")

# Print the overall results to the console
print("Overall Group Detection Results")
print("===============================")
print(f"Total Groups: {total_groups}")
print(f"Correctly Detected Groups: {correct_groups}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print(f"Results saved to {output_file}")

# --- Heatmap of group detection by district and year for XGBoost ---
heatmap_data = []
for (district, seasonYear), group in grouped_results:
    predicted_counts = group['Predicted_Label'].value_counts()
    count_0 = predicted_counts.get(0, 0)
    count_1 = predicted_counts.get(1, 0)
    majority_prediction = 1 if count_1 >= count_0 else 0
    actual_label = group['Actual_Label'].mode()[0]
    correct = int(majority_prediction == actual_label)
    heatmap_data.append({'District': district, 'Year': seasonYear, 'Correct': correct})

heatmap_df = pd.DataFrame(heatmap_data)
heatmap_matrix = heatmap_df.pivot(index='District', columns='Year', values='Correct')

plt.figure(figsize=(10, 4))
sns.heatmap(heatmap_matrix, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Correct Group Detection (1=Correct, 0=Wrong)'})
plt.title('XGBoost Group Detection Accuracy by District and Year')
plt.xlabel('Year')
plt.ylabel('District')
plt.tight_layout()
plt.savefig("MoreData/ResultsWeightedSum/group_detection_heatmap_xgb.png")
plt.clf()

# SHAP Analysis for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Save SHAP summary plot as an image file
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_test.columns)
plt.savefig("MoreData/ResultsWeightedSum/shap_summary_bar_plot_xgb.png")  # Save plot to file
plt.clf()  # Clear the current plot

# Save the SHAP summary plot as an image
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
plt.savefig("MoreData/ResultsWeightedSum/shap_summary_plot_xgb.png")  # Save plot to file
plt.clf()  # Clear the current plot

# ROC and AUC Curves for XGBoost
y_scores_xgb = xgb_model.predict_proba(X_test)[:, 1]
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_scores_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Create figure for ROC curve
plt.figure(figsize=(10, 8))

# Plot ROC curve
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc_xgb:.2f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Customize plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC Curves - XGBoost')
plt.legend(loc="lower right")

# Add AUC score text
plt.text(0.6, 0.2, f'AUC Score: {roc_auc_xgb:.4f}', 
         bbox=dict(facecolor='white', alpha=0.8))

# Save plot
plt.savefig("MoreData/ResultsWeightedSum/roc_auc_curve_xgb.png", 
            bbox_inches='tight', dpi=300)
plt.clf()

# Save AUC score to performance file
with open(performance_output_file, "a") as f:
    f.write("\nXGBoost ROC-AUC Results\n")
    f.write("=====================\n")
    f.write(f"AUC Score: {roc_auc_xgb:.4f}\n\n")

## *****************************************************************************************************************##

### Random Forest Classifier ###
model_names.append('Random Forest')
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Train-Test Split Results
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_rf)
test_precision = precision_score(y_test, y_pred_rf)
test_recall = recall_score(y_test, y_pred_rf)

# Cross-validation Results
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), 
                               X_train, y_train, cv=cv, scoring='accuracy')
cv_precisions = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), 
                               X_train, y_train, cv=cv, scoring='precision')
cv_recalls = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), 
                            X_train, y_train, cv=cv, scoring='recall')

# Calculate means for storing
accuracy_rf = cv_accuracies.mean()
precision_rf = cv_precisions.mean()
recall_rf = cv_recalls.mean()

accuracies.append(test_accuracy)
precisions.append(test_precision)
recalls.append(test_recall)

# Print and save results
print("\nRandom Forest Results")
print("====================")
print("Cross-validation Results:")
print(f"CV Accuracy: {accuracy_rf:.4f} ± {cv_accuracies.std():.4f}")
print(f"CV Precision: {precision_rf:.4f} ± {cv_precisions.std():.4f}")
print(f"CV Recall: {recall_rf:.4f} ± {cv_recalls.std():.4f}")
print("\nTrain-Test Split Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Save results to file
with open(performance_output_file, "a") as f:
    f.write("\nRandom Forest Results\n")
    f.write("====================\n\n")
    f.write("Cross-validation Results:\n")
    f.write(f"CV Accuracy: {accuracy_rf:.4f} ± {cv_accuracies.std():.4f}\n")
    f.write(f"CV Precision: {precision_rf:.4f} ± {cv_precisions.std():.4f}\n")
    f.write(f"CV Recall: {recall_rf:.4f} ± {cv_recalls.std():.4f}\n\n")
    f.write("Train-Test Split Results:\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n\n")

# Create and save a summary table
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Cross-validation': [
        f"{accuracy_rf:.4f} ± {cv_accuracies.std():.4f}",
        f"{precision_rf:.4f} ± {cv_precisions.std():.4f}",
        f"{recall_rf:.4f} ± {cv_recalls.std():.4f}"
    ],
    'Train-Test Split': [
        f"{test_accuracy:.4f}",
        f"{test_precision:.4f}",
        f"{test_recall:.4f}"
    ]
})

# Save metrics table as image
plt.figure(figsize=(10, 4))
ax = plt.gca()
ax.axis('tight')
ax.axis('off')
table = plt.table(cellText=metrics_df.values,
                 colLabels=metrics_df.columns,
                 rowLabels=None,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title("Random Forest Performance Metrics Comparison")
plt.savefig("MoreData/ResultsWeightedSum/random_forest_metrics_comparison.png", 
            bbox_inches='tight', dpi=300)
plt.close()

# Prepare the output file
results_df["Predicted_Label"] = y_pred_rf
output_file = "MoreData/ResultsWeightedSum/rf_test_results.txt"
correct_groups = 0
total_groups = len(grouped_results)

with open(output_file, "w") as f:
    for (district, seasonYear), group in grouped_results:
        f.write(f"District: {district}, SeasonYear: {seasonYear} \n")
        f.write(f"Total Test Cases: {len(group)}\n")
       
        # Perform voting on the predicted labels
        predicted_counts = group['Predicted_Label'].value_counts()
        count_0 = predicted_counts.get(0, 0)
        count_1 = predicted_counts.get(1, 0)

         # Determine the majority prediction
        majority_prediction = 1 if count_1 >= count_0 else 0

        # Check if the majority prediction matches the actual label
        actual_label = group['Actual_Label'].mode()[0]
        if majority_prediction == actual_label:
            f.write("Group Detected Correctly\n")
            correct_groups += 1
        else:
            f.write("Group Detected Wrongly\n")

        # Write the voting results
        f.write(f"Predicted 0: {count_0}, Predicted 1: {count_1}\n")

        f.write("Actual vs Predicted:\n")
        for idx, row in group.iterrows():
            f.write(f"Actual: {row['Actual_Label']}, Predicted: {row['Predicted_Label']}\n")
        f.write("\n")

# Calculate accuracy, precision, and recall for the groups
accuracy = correct_groups / total_groups
precision = correct_groups / (correct_groups + (total_groups - correct_groups))
recall = correct_groups / total_groups

# Save the overall results to the file
with open(output_file, "a") as f:
    f.write("Overall Group Detection Results\n")
    f.write("===============================\n")
    f.write(f"Total Groups: {total_groups}\n")
    f.write(f"Correctly Detected Groups: {correct_groups}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")

with open(performance_output_file, "a") as f:
    f.write("Overall Group Detection Results\n")
    f.write("===============================\n")
    f.write(f"Total Groups: {total_groups}\n")
    f.write(f"Correctly Detected Groups: {correct_groups}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n\n")

# Print the overall results to the console
print("Overall Group Detection Results")
print("===============================")
print(f"Total Groups: {total_groups}")
print(f"Correctly Detected Groups: {correct_groups}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print(f"Results saved to {output_file}")

# --- Heatmap of group detection by district and year for Random Forest ---
heatmap_data_rf = []
for (district, seasonYear), group in grouped_results:
    predicted_counts = group['Predicted_Label'].value_counts()
    count_0 = predicted_counts.get(0, 0)
    count_1 = predicted_counts.get(1, 0)
    majority_prediction = 1 if count_1 >= count_0 else 0
    actual_label = group['Actual_Label'].mode()[0]
    correct = int(majority_prediction == actual_label)
    heatmap_data_rf.append({'District': district, 'Year': seasonYear, 'Correct': correct})

heatmap_df_rf = pd.DataFrame(heatmap_data_rf)
heatmap_matrix_rf = heatmap_df_rf.pivot(index='District', columns='Year', values='Correct')

plt.figure(figsize=(10, 4))
sns.heatmap(heatmap_matrix_rf, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Correct Group Detection (1=Correct, 0=Wrong)'})
plt.title('Random Forest Group Detection Accuracy by District and Year')
plt.xlabel('Year')
plt.ylabel('District')
plt.tight_layout()
plt.savefig("MoreData/ResultsWeightedSum/group_detection_heatmap_rf.png")
plt.clf()

# SHAP Analysis with TreeExplainer for Random Forest
explainer_rf = shap.TreeExplainer(rf)  # 'rf' is your trained Random Forest model
shap_values_rf = explainer_rf.shap_values(X_test)

print(shap_values_rf.shape)
print(X_test.columns)

# Use SHAP values for the positive class (assuming drought is labeled as 1)
shap_values_rf_drought = shap_values_rf[:, :, 1] 

# Plot SHAP summary plot as a bar chart to show feature importance
shap.summary_plot(shap_values_rf_drought, X_test, plot_type="bar", feature_names=X_test.columns)
plt.savefig("MoreData/ResultsWeightedSum/shap_summary_bar_plot_rf.png")  # Save plot to file
plt.clf()  # Clear the current plot

# Plot the full SHAP summary plot to visualize feature impact on individual predictions
shap.summary_plot(shap_values_rf_drought, X_test, feature_names=X_test.columns)
plt.savefig("MoreData/ResultsWeightedSum/shap_summary_plot_rf.png")  # Save plot to file
plt.clf()  # Clear the current plot

# ROC Curve for Random Forest
y_scores_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_scores_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Create figure for ROC curve
plt.figure(figsize=(10, 8))

# Plot ROC curve
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc_rf:.2f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Customize plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC Curves - Random Forest')
plt.legend(loc="lower right")

# Add AUC score text
plt.text(0.6, 0.2, f'AUC Score: {roc_auc_rf:.4f}', 
         bbox=dict(facecolor='white', alpha=0.8))

# Save plot
plt.savefig("MoreData/ResultsWeightedSum/roc_auc_curve_rf.png", 
            bbox_inches='tight', dpi=300)
plt.clf()

# Save AUC score to performance file
with open(performance_output_file, "a") as f:
    f.write("\nRandom Forest ROC-AUC Results\n")
    f.write("===========================\n")
    f.write(f"AUC Score: {roc_auc_rf:.4f}\n\n")

## *****************************************************************************************************************##

# Bagging Classifier
### Bagging Classifier ###
model_names.append('Bagging')
bagging = BaggingClassifier(n_estimators=200, random_state=42)

# Train-Test Split Results
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_bagging)
test_precision = precision_score(y_test, y_pred_bagging)
test_recall = recall_score(y_test, y_pred_bagging)

# Cross-validation Results
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = cross_val_score(BaggingClassifier(n_estimators=200, random_state=42), 
                               X_train, y_train, cv=cv, scoring='accuracy')
cv_precisions = cross_val_score(BaggingClassifier(n_estimators=200, random_state=42), 
                               X_train, y_train, cv=cv, scoring='precision')
cv_recalls = cross_val_score(BaggingClassifier(n_estimators=200, random_state=42), 
                            X_train, y_train, cv=cv, scoring='recall')

# Calculate means for storing
accuracy_bagging = cv_accuracies.mean()
precision_bagging = cv_precisions.mean()
recall_bagging = cv_recalls.mean()

accuracies.append(test_accuracy)
precisions.append(test_precision)
recalls.append(test_recall)

# Print and save results
print("\nBagging Classifier Results")
print("========================")
print("Cross-validation Results:")
print(f"CV Accuracy: {accuracy_bagging:.4f} ± {cv_accuracies.std():.4f}")
print(f"CV Precision: {precision_bagging:.4f} ± {cv_precisions.std():.4f}")
print(f"CV Recall: {recall_bagging:.4f} ± {cv_recalls.std():.4f}")
print("\nTrain-Test Split Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Save results to file
with open(performance_output_file, "a") as f:
    f.write("\nBagging Classifier Results\n")
    f.write("========================\n\n")
    f.write("Cross-validation Results:\n")
    f.write(f"CV Accuracy: {accuracy_bagging:.4f} ± {cv_accuracies.std():.4f}\n")
    f.write(f"CV Precision: {precision_bagging:.4f} ± {cv_precisions.std():.4f}\n")
    f.write(f"CV Recall: {recall_bagging:.4f} ± {cv_recalls.std():.4f}\n\n")
    f.write("Train-Test Split Results:\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n\n")

# Create and save a summary table
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Cross-validation': [
        f"{accuracy_bagging:.4f} ± {cv_accuracies.std():.4f}",
        f"{precision_bagging:.4f} ± {cv_precisions.std():.4f}",
        f"{recall_bagging:.4f} ± {cv_recalls.std():.4f}"
    ],
    'Train-Test Split': [
        f"{test_accuracy:.4f}",
        f"{test_precision:.4f}",
        f"{test_recall:.4f}"
    ]
})

# Save metrics table as image
plt.figure(figsize=(10, 4))
ax = plt.gca()
ax.axis('tight')
ax.axis('off')
table = plt.table(cellText=metrics_df.values,
                 colLabels=metrics_df.columns,
                 rowLabels=None,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title("Bagging Classifier Performance Metrics Comparison")
plt.savefig("MoreData/ResultsWeightedSum/bagging_metrics_comparison.png", 
            bbox_inches='tight', dpi=300)
plt.close()

# Continue with existing group detection code...
# Prepare the output file
results_df["Predicted_Label"] = y_pred_bagging
output_file = "MoreData/ResultsWeightedSum/bagging_test_results.txt"
correct_groups = 0
total_groups = len(grouped_results)

with open(output_file, "w") as f:
    for (district, seasonYear), group in grouped_results:
        f.write(f"District: {district}, SeasonYear: {seasonYear} \n")
        f.write(f"Total Test Cases: {len(group)}\n")
       
        # Perform voting on the predicted labels
        predicted_counts = group['Predicted_Label'].value_counts()
        count_0 = predicted_counts.get(0, 0)
        count_1 = predicted_counts.get(1, 0)

         # Determine the majority prediction
        majority_prediction = 1 if count_1 >= count_0 else 0

        # Check if the majority prediction matches the actual label
        actual_label = group['Actual_Label'].mode()[0]
        if majority_prediction == actual_label:
            f.write("Group Detected Correctly\n")
            correct_groups += 1
        else:
            f.write("Group Detected Wrongly\n")

        # Write the voting results
        f.write(f"Predicted 0: {count_0}, Predicted 1: {count_1}\n")

        f.write("Actual vs Predicted:\n")
        for idx, row in group.iterrows():
            f.write(f"Actual: {row['Actual_Label']}, Predicted: {row['Predicted_Label']}\n")
        f.write("\n")

# Calculate accuracy, precision, and recall for the groups
accuracy = correct_groups / total_groups
precision = correct_groups / (correct_groups + (total_groups - correct_groups))
recall = correct_groups / total_groups

# Save the overall results to the file
with open(output_file, "a") as f:
    f.write("Overall Group Detection Results\n")
    f.write("===============================\n")
    f.write(f"Total Groups: {total_groups}\n")
    f.write(f"Correctly Detected Groups: {correct_groups}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")

with open(performance_output_file, "a") as f:
    f.write("Overall Group Detection Results\n")
    f.write("===============================\n")
    f.write(f"Total Groups: {total_groups}\n")
    f.write(f"Correctly Detected Groups: {correct_groups}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n\n")

# Print the overall results to the console
print("Overall Group Detection Results")
print("===============================")
print(f"Total Groups: {total_groups}")
print(f"Correctly Detected Groups: {correct_groups}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print(f"Results saved to {output_file}")

# --- Heatmap of group detection by district and year for Bagging ---
heatmap_data_bagging = []
for (district, seasonYear), group in grouped_results:
    predicted_counts = group['Predicted_Label'].value_counts()
    count_0 = predicted_counts.get(0, 0)
    count_1 = predicted_counts.get(1, 0)
    majority_prediction = 1 if count_1 >= count_0 else 0
    actual_label = group['Actual_Label'].mode()[0]
    correct = int(majority_prediction == actual_label)
    heatmap_data_bagging.append({'District': district, 'Year': seasonYear, 'Correct': correct})

heatmap_df_bagging = pd.DataFrame(heatmap_data_bagging)
heatmap_matrix_bagging = heatmap_df_bagging.pivot(index='District', columns='Year', values='Correct')

plt.figure(figsize=(10, 4))
sns.heatmap(heatmap_matrix_bagging, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Correct Group Detection (1=Correct, 0=Wrong)'})
plt.title('Bagging Group Detection Accuracy by District and Year')
plt.xlabel('Year')
plt.ylabel('District')
plt.tight_layout()
plt.savefig("MoreData/ResultsWeightedSum/group_detection_heatmap_bagging.png")
plt.clf()

# Use PermutationExplainer for the BaggingClassifier
explainer_bagging = shap.PermutationExplainer(bagging.predict, X_test)
shap_values_bagging = explainer_bagging.shap_values(X_test)

# Plot SHAP summary plot as a bar chart for feature importance
shap.summary_plot(shap_values_bagging, X_test, plot_type="bar", feature_names=X_test.columns)
plt.savefig("MoreData/ResultsWeightedSum/shap_summary_bar_plot_bagging.png")  # Save bar plot to file
plt.clf()  # Clear the current plot

# Plot the full SHAP summary plot to visualize feature impact on individual predictions
shap.summary_plot(shap_values_bagging, X_test, feature_names=X_test.columns)
plt.savefig("MoreData/ResultsWeightedSum/shap_summary_plot_bagging.png")  # Save beeswarm plot to file
plt.clf()  # Clear the current plot

# ROC Curve for Bagging Classifier
y_scores_bagging = bagging.predict_proba(X_test)[:, 1]
fpr_bagging, tpr_bagging, _ = roc_curve(y_test, y_scores_bagging)
roc_auc_bagging = auc(fpr_bagging, tpr_bagging)

# Create figure for ROC curve
plt.figure(figsize=(10, 8))

# Plot ROC curve
plt.plot(fpr_bagging, tpr_bagging, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc_bagging:.2f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Customize plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC Curves - Bagging')
plt.legend(loc="lower right")

# Add AUC score text
plt.text(0.6, 0.2, f'AUC Score: {roc_auc_bagging:.4f}', 
         bbox=dict(facecolor='white', alpha=0.8))

# Save plot
plt.savefig("MoreData/ResultsWeightedSum/roc_auc_curve_bagging.png", 
            bbox_inches='tight', dpi=300)
plt.clf()

# Save AUC score to performance file
with open(performance_output_file, "a") as f:
    f.write("\nBagging ROC-AUC Results\n")
    f.write("=====================\n")
    f.write(f"AUC Score: {roc_auc_bagging:.4f}\n\n")

# --- End of Bagging Classifier Code ---
## *****************************************************************************************************************##

### Gradient Boosting Classifier ###
model_names.append('Gradient Boosting')
gb = GradientBoostingClassifier(n_estimators=200,random_state=42)

# Train-Test Split Results
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_gb)
test_precision = precision_score(y_test, y_pred_gb)
test_recall = recall_score(y_test, y_pred_gb)

# Cross-validation Results
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = cross_val_score(GradientBoostingClassifier(n_estimators=200,random_state=42), 
                               X_train, y_train, cv=cv, scoring='accuracy')
cv_precisions = cross_val_score(GradientBoostingClassifier(n_estimators=200,random_state=42), 
                               X_train, y_train, cv=cv, scoring='precision')
cv_recalls = cross_val_score(GradientBoostingClassifier(n_estimators=200,random_state=42), 
                            X_train, y_train, cv=cv, scoring='recall')

# Calculate means for storing
accuracy_gb = cv_accuracies.mean()
precision_gb = cv_precisions.mean()
recall_gb = cv_recalls.mean()

accuracies.append(test_accuracy)
precisions.append(test_precision)
recalls.append(test_recall)

# Print and save results
print("\nGradient Boosting Results")
print("========================")
print("Cross-validation Results:")
print(f"CV Accuracy: {accuracy_gb:.4f} ± {cv_accuracies.std():.4f}")
print(f"CV Precision: {precision_gb:.4f} ± {cv_precisions.std():.4f}")
print(f"CV Recall: {recall_gb:.4f} ± {cv_recalls.std():.4f}")
print("\nTrain-Test Split Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Save results to file
with open(performance_output_file, "a") as f:
    f.write("\nGradient Boosting Results\n")
    f.write("========================\n\n")
    f.write("Cross-validation Results:\n")
    f.write(f"CV Accuracy: {accuracy_gb:.4f} ± {cv_accuracies.std():.4f}\n")
    f.write(f"CV Precision: {precision_gb:.4f} ± {cv_precisions.std():.4f}\n")
    f.write(f"CV Recall: {recall_gb:.4f} ± {cv_recalls.std():.4f}\n\n")
    f.write("Train-Test Split Results:\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n\n")

# Create and save a summary table
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Cross-validation': [
        f"{accuracy_gb:.4f} ± {cv_accuracies.std():.4f}",
        f"{precision_gb:.4f} ± {cv_precisions.std():.4f}",
        f"{recall_gb:.4f} ± {cv_recalls.std():.4f}"
    ],
    'Train-Test Split': [
        f"{test_accuracy:.4f}",
        f"{test_precision:.4f}",
        f"{test_recall:.4f}"
    ]
})

# Save metrics table as image
plt.figure(figsize=(10, 4))
ax = plt.gca()
ax.axis('tight')
ax.axis('off')
table = plt.table(cellText=metrics_df.values,
                 colLabels=metrics_df.columns,
                 rowLabels=None,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title("Gradient Boosting Performance Metrics Comparison")
plt.savefig("MoreData/ResultsWeightedSum/gradient_boosting_metrics_comparison.png", 
            bbox_inches='tight', dpi=300)
plt.close()

# Prepare the output file
results_df["Predicted_Label"] = y_pred_gb
output_file = "MoreData/ResultsWeightedSum/gb_test_results.txt"
correct_groups = 0
total_groups = len(grouped_results)

with open(output_file, "w") as f:
    for (district, seasonYear), group in grouped_results:
        f.write(f"District: {district}, SeasonYear: {seasonYear} \n")
        f.write(f"Total Test Cases: {len(group)}\n")
       
        # Perform voting on the predicted labels
        predicted_counts = group['Predicted_Label'].value_counts()
        count_0 = predicted_counts.get(0, 0)
        count_1 = predicted_counts.get(1, 0)

         # Determine the majority prediction
        majority_prediction = 1 if count_1 >= count_0 else 0

        # Check if the majority prediction matches the actual label
        actual_label = group['Actual_Label'].mode()[0]
        if majority_prediction == actual_label:
            f.write("Group Detected Correctly\n")
            correct_groups += 1
        else:
            f.write("Group Detected Wrongly\n")

        # Write the voting results
        f.write(f"Predicted 0: {count_0}, Predicted 1: {count_1}\n")

        f.write("Actual vs Predicted:\n")
        for idx, row in group.iterrows():
            f.write(f"Actual: {row['Actual_Label']}, Predicted: {row['Predicted_Label']}\n")
        f.write("\n")

# Calculate accuracy, precision, and recall for the groups
accuracy = correct_groups / total_groups
precision = correct_groups / (correct_groups + (total_groups - correct_groups))
recall = correct_groups / total_groups

# Save the overall results to the file
with open(output_file, "a") as f:
    f.write("Overall Group Detection Results\n")
    f.write("===============================\n")
    f.write(f"Total Groups: {total_groups}\n")
    f.write(f"Correctly Detected Groups: {correct_groups}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")

with open(performance_output_file, "a") as f:
    f.write("Overall Group Detection Results\n")
    f.write("===============================\n")
    f.write(f"Total Groups: {total_groups}\n")
    f.write(f"Correctly Detected Groups: {correct_groups}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n\n")

# Print the overall results to the console
print("Overall Group Detection Results")
print("===============================")
print(f"Total Groups: {total_groups}")
print(f"Correctly Detected Groups: {correct_groups}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print(f"Results saved to {output_file}")

# --- Heatmap of group detection by district and year for Gradient Boosting ---
heatmap_data_gb = []
for (district, seasonYear), group in grouped_results:
    predicted_counts = group['Predicted_Label'].value_counts()
    count_0 = predicted_counts.get(0, 0)
    count_1 = predicted_counts.get(1, 0)
    majority_prediction = 1 if count_1 >= count_0 else 0
    actual_label = group['Actual_Label'].mode()[0]
    correct = int(majority_prediction == actual_label)
    heatmap_data_gb.append({'District': district, 'Year': seasonYear, 'Correct': correct})

heatmap_df_gb = pd.DataFrame(heatmap_data_gb)
heatmap_matrix_gb = heatmap_df_gb.pivot(index='District', columns='Year', values='Correct')

plt.figure(figsize=(10, 4))
sns.heatmap(heatmap_matrix_gb, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Correct Group Detection (1=Correct, 0=Wrong)'})
plt.title('Gradient Boosting Group Detection Accuracy by District and Year')
plt.xlabel('Year')
plt.ylabel('District')
plt.tight_layout()
plt.savefig("MoreData/ResultsWeightedSum/group_detection_heatmap_gb.png")
plt.clf()

# SHAP Analysis for Gradient Boosting
explainer_gb = shap.TreeExplainer(gb)
shap_values_gb = explainer_gb.shap_values(X_test)

# Save SHAP summary plot as an image file for Gradient Boosting
shap.summary_plot(shap_values_gb, X_test, plot_type="bar", feature_names=X_test.columns)
plt.savefig("MoreData/ResultsWeightedSum/shap_summary_bar_plot_gb.png")  # Save plot to file
plt.clf()  # Clear the current plot

# Save the SHAP summary plot as an image for Gradient Boosting
shap.summary_plot(shap_values_gb, X_test, feature_names=X_test.columns)
plt.savefig("MoreData/ResultsWeightedSum/shap_summary_plot_gb.png")  # Save plot to file
plt.clf()  # Clear the current plot

# ROC Curve for Gradient Boosting
y_scores_gb = gb.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_scores_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

# Create figure for ROC curve
plt.figure(figsize=(10, 8))

# Plot ROC curve
plt.plot(fpr_gb, tpr_gb, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc_gb:.2f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Customize plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC Curves - Gradient Boosting')
plt.legend(loc="lower right")

# Add AUC score text
plt.text(0.6, 0.2, f'AUC Score: {roc_auc_gb:.4f}', 
         bbox=dict(facecolor='white', alpha=0.8))

# Save plot
plt.savefig("MoreData/ResultsWeightedSum/roc_auc_curve_gb.png", 
            bbox_inches='tight', dpi=300)
plt.clf()

# Save AUC score to performance file
with open(performance_output_file, "a") as f:
    f.write("\nGradient Boosting ROC-AUC Results\n")
    f.write("==============================\n")
    f.write(f"AUC Score: {roc_auc_gb:.4f}\n\n")

## *****************************************************************************************************************##
# Combined ROC Curve for all models
plt.figure(figsize=(12, 8))

# Plot ROC curves for each model with different colors
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, 
         label=f'XGBoost (AUC = {roc_auc_xgb:.4f})')
plt.plot(fpr_rf, tpr_rf, color='forestgreen', lw=2, 
         label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
plt.plot(fpr_bagging, tpr_bagging, color='royalblue', lw=2, 
         label=f'Bagging (AUC = {roc_auc_bagging:.4f})')
plt.plot(fpr_gb, tpr_gb, color='darkred', lw=2, 
         label=f'Gradient Boosting (AUC = {roc_auc_gb:.4f})')

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

# Customize plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves for All Models')
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Save plot with higher resolution
plt.savefig("MoreData/ResultsWeightedSum/combined_roc_curves.png", 
            bbox_inches='tight', dpi=300)
plt.clf()

# Save AUC scores comparison to performance file
with open(performance_output_file, "a") as f:
    f.write("\nROC-AUC Comparison for All Models\n")
    f.write("================================\n")
    f.write(f"XGBoost AUC Score: {roc_auc_xgb:.4f}\n")
    f.write(f"Random Forest AUC Score: {roc_auc_rf:.4f}\n")
    f.write(f"Bagging AUC Score: {roc_auc_bagging:.4f}\n")
    f.write(f"Gradient Boosting AUC Score: {roc_auc_gb:.4f}\n\n")

## *****************************************************************************************************************##
# Create a DataFrame with the metrics
metrics_df1 = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls
})

# Create a table plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=metrics_df1.values, colLabels=metrics_df1.columns, cellLoc='center', loc='center')

# Save the table as a PNG image
plt.savefig('MoreData/ResultsWeightedSum/model_performance_table.png')
plt.clf()  # Clear the current plot
## *****************************************************************************************************************##

# Get SHAP values for each model
shap_values_rf_drought_mean = np.abs(shap_values_rf_drought).mean(axis=0)
shap_values_xgb_mean = np.abs(shap_values).mean(axis=0)
shap_values_gb_mean = np.abs(shap_values_gb).mean(axis=0)
shap_values_bagging_mean = np.abs(shap_values_bagging).mean(axis=0)

# Convert the SHAP values into DataFrames to use in the weighted sum
rf_importance = pd.Series(shap_values_rf_drought_mean, index=X_test.columns)
xgb_importance = pd.Series(shap_values_xgb_mean, index=X_test.columns)
gb_importance = pd.Series(shap_values_gb_mean, index=X_test.columns)
bagging_importance = pd.Series(shap_values_bagging_mean, index=X_test.columns)

## *****************************************************************************************************************##

# Function to calculate the weighted sum of feature importance
def weighted_sum_rankings(weights, *importance_scores):
    # Initialize an array to store the weighted sum for each feature
    features = importance_scores[0].index  # Assuming all importance_scores have the same features
    weighted_sum = {feature: 0 for feature in features}

    # Iterate over the importance scores and weights, compute the weighted sum
    for importance, weight in zip(importance_scores, weights):
        weighted_importance = importance * weight  # Multiply importance by weight
        for feature, score in weighted_importance.items():
            weighted_sum[feature] += score  # Sum the weighted scores for each feature

    # Create a DataFrame with the weighted sum for each feature
    weighted_sum_df = pd.DataFrame(list(weighted_sum.items()), columns=['Feature', 'Weighted Sum'])
    weighted_sum_df = weighted_sum_df.sort_values(by='Weighted Sum', ascending=False)

    return weighted_sum_df


# Assign weights based on the model accuracies (we can use any other criterion here)
#weights = [accuracy_xgb, accuracy_bagging, accuracy_rf, accuracy_gb]
weights = [accuracy_xgb, accuracy_bagging, accuracy_rf]

# Apply weighted sum to get the top 5 features
#weighted_sum_df = weighted_sum_rankings(weights, xgb_importance, bagging_importance, rf_importance, gb_importance)
weighted_sum_df = weighted_sum_rankings(weights, xgb_importance, bagging_importance, rf_importance)

# Print the top 5 features based on weighted sum
print(weighted_sum_df.head(5))

# Plot the top 5 features based on weighted sum
top_5_features_weighted_sum = weighted_sum_df.head(5)
plt.figure(figsize=(10, 6))
plt.barh(top_5_features_weighted_sum['Feature'], top_5_features_weighted_sum['Weighted Sum'], color='lightgreen')
plt.xlabel('Weighted Sum of Feature Importance')
plt.title('Top 5 Features Based on Weighted Sum of Importance')
plt.gca().invert_yaxis()  # Invert the y-axis to display the top feature on top
plt.tight_layout()

# Save the plot
plt.savefig("MoreData/ResultsWeightedSum/top_5_features_weighted_sum.png")
plt.clf()  # Clear the plot

top_5_features_list = weighted_sum_df.head(5)['Feature'].tolist()

## *****************************************************************************************************************##

def evaluate_model_with_top_features(model, X_train, X_test, y_train, y_test, top_features):
    """
    Evaluate model using both cross-validation and train-test split with top features
    """
    # Use only the top features
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    
    # Cross-validation Results
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X_train_top, y_train, cv=cv, scoring='accuracy')
    prec_scores = cross_val_score(model, X_train_top, y_train, cv=cv, scoring='precision')
    rec_scores = cross_val_score(model, X_train_top, y_train, cv=cv, scoring='recall')

    # Train-Test Split Results
    model.fit(X_train_top, y_train)
    y_pred = model.predict(X_test_top)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    # Return both CV and Test results
    results = {
        'cv_accuracy': f"{acc_scores.mean():.4f} ± {acc_scores.std():.4f}",
        'cv_precision': f"{prec_scores.mean():.4f} ± {prec_scores.std():.4f}",
        'cv_recall': f"{rec_scores.mean():.4f} ± {rec_scores.std():.4f}",
        'test_accuracy': f"{test_accuracy:.4f}",
        'test_precision': f"{test_precision:.4f}",
        'test_recall': f"{test_recall:.4f}"
    }
    
    return results

# Evaluate each model using the top 1 to top 5 features
metrics = {}
for model_name, model in {
    'XGBoost': xgb_model,
    'Random Forest': rf,
    'Bagging': bagging,
    'Gradient Boosting': gb
}.items():
    metrics[model_name] = []
    for i in range(1, 6):
        top_features = top_5_features_list[:i]
        results = evaluate_model_with_top_features(
            model, X_train, X_test, y_train, y_test, top_features
        )
        metrics[model_name].append(results)

# Create separate DataFrames for CV and Test results
cv_results = pd.DataFrame(columns=['Model', 'Features', 'Accuracy', 'Precision', 'Recall'])
test_results = pd.DataFrame(columns=['Model', 'Features', 'Accuracy', 'Precision', 'Recall'])

for model_name in metrics:
    for i, result in enumerate(metrics[model_name], 1):
        # For CV results
        cv_results = pd.concat([cv_results, pd.DataFrame([{
            'Model': model_name,
            'Features': f'Top {i}',
            'Accuracy': result['cv_accuracy'],
            'Precision': result['cv_precision'],
            'Recall': result['cv_recall']
        }])], ignore_index=True)
        
        # For Test results
        test_results = pd.concat([test_results, pd.DataFrame([{
            'Model': model_name,
            'Features': f'Top {i}',
            'Accuracy': result['test_accuracy'],
            'Precision': result['test_precision'],
            'Recall': result['test_recall']
        }])], ignore_index=True)

# Save results to file and create visualizations
with open("MoreData/ResultsWeightedSum/top_features_performance.txt", "w") as f:
    f.write("Cross-validation Results\n")
    f.write("=======================\n")
    f.write(cv_results.to_string())
    f.write("\n\nTest Set Results\n")
    f.write("================\n")
    f.write(test_results.to_string())

# Create and save visualization for CV results
plt.figure(figsize=(15, 8))
ax = plt.gca()
ax.axis('tight')
ax.axis('off')
cv_table = plt.table(cellText=cv_results.values,
                    colLabels=cv_results.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
cv_table.auto_set_font_size(False)
cv_table.set_fontsize(9)
cv_table.scale(1.2, 1.5)
plt.title("Cross-validation Performance with Top Features")
plt.savefig("MoreData/ResultsWeightedSum/cv_performance_top_features.png", 
            bbox_inches='tight', dpi=300)
plt.close()

# Create and save visualization for Test results
plt.figure(figsize=(15, 8))
ax = plt.gca()
ax.axis('tight')
ax.axis('off')
test_table = plt.table(cellText=test_results.values,
                      colLabels=test_results.columns,
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
test_table.auto_set_font_size(False)
test_table.set_fontsize(9)
test_table.scale(1.2, 1.5)
plt.title("Test Set Performance with Top Features")
plt.savefig("MoreData/ResultsWeightedSum/test_performance_top_features.png", 
            bbox_inches='tight', dpi=300)
plt.close()

## *****************************************************************************************************************##
## Error Analysis with confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

f = open(performance_output_file, "a")

# Function to plot confusion matrix and save as an image
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{cm}\n")  # Print the confusion matrix
    f.write(f"Confusion Matrix for {model_name}:\n{cm}\n")  

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Drought (0)', 'Drought (1)'], yticklabels=['No Drought (0)', 'Drought (1)'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"MoreData/ResultsWeightedSum/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")  # Save confusion matrix as an image
    plt.clf()  # Clear the current plot

# Evaluate and perform error analysis for each model
models = {
    "XGBoost": xgb_model,
    "Random Forest": rf,
    "Bagging": bagging,
    "Gradient Boosting": gb
}

for model_name, model in models.items():
    # Get predictions for each model
    if model_name == "XGBoost":
        y_pred = y_pred_xgb
    elif model_name == "Random Forest":
        y_pred = y_pred_rf
    elif model_name == "Bagging":
        y_pred = y_pred_bagging
    elif model_name == "Gradient Boosting":
        y_pred = y_pred_gb
    
    # Plot confusion matrix and save as an image
    plot_confusion_matrix(y_test, y_pred, model_name)
    
f.close()
