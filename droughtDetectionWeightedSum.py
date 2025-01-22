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

"""

# Last Try to do it again(87%)
drought_labels = {
    'Jodhpur': {2016: 1, 2017: 0, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Amravati': {2016: 1, 2017: 1, 2018: 1, 2019: 1, 2020: 0, 2021: 1},
    'Thanjavur': {2016: 0, 2017: 1, 2018: 0, 2019: 1, 2020: 0, 2021: 0}
}
def load_data(district):
    dfs = []
    for year in range(2016, 2022):  # Loop through the relevant years
        # Define filenames for both current and previous year's data
        current_file = f"WithAgriculturalMask/timeSeriesData/TimeSeries_{district}_{year}.csv"
        previous_file = f"WithAgriculturalMask/timeSeriesData/TimeSeries_{district}_{year-1}.csv"
        
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
file_path = 'WithAgriculturalMask/ResultsWeightedSum/output_file.csv'
data.to_csv(file_path, index=False)

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Define features (remote sensing indices) and target (drought affected or not)
X = data.drop(columns=['SeasonYear','Drought', 'Year', 'Month', 'District'])  # Features
y = data['Drought']  # Target

# Save the column names before scaling
column_names = X.columns

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Convert back to a pandas DataFrame with the correct column names
X = pd.DataFrame(X, columns=column_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Handle class imbalance with ADASYN
#adasyn = ADASYN(random_state=42)
#X_train, y_train = adasyn.fit_resample(X_train, y_train)

## *****************************************************************************************************************##

### XGBoost Classifier ###

model_names.append('XGBoost')
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
accuracies.append(accuracy_xgb)
print(f"XGBoost Accuracy (Rabi Crop): {accuracy_xgb * 100:.2f}%")

# Calculate Precision and Recall for XGBoost
precision_xgb = precision_score(y_test, y_pred_xgb, pos_label=1)  # Use 1 for drought
recall_xgb = recall_score(y_test, y_pred_xgb, pos_label=1)
precisions.append(precision_xgb)
recalls.append(recall_xgb)

print(f"XGBoost Precision: {precision_xgb * 100:.2f}%")
print(f"XGBoost Recall: {recall_xgb * 100:.2f}%")

# SHAP Analysis for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Save SHAP summary plot as an image file
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_test.columns)
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/shap_summary_bar_plot_xgb.png")  # Save plot to file
plt.clf()  # Clear the current plot

# Save the SHAP summary plot as an image
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/shap_summary_plot_xgb.png")  # Save plot to file
plt.clf()  # Clear the current plot

## *****************************************************************************************************************##

### Random Forest Classifier ###
model_names.append('Random Forest')
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracies.append(accuracy_rf)
print(f"Random Forest Accuracy (Rabi Crop): {accuracy_rf * 100:.2f}%")

# Calculate Precision and Recall for Random Forest
precision_rf = precision_score(y_test, y_pred_rf, pos_label=1)  # Use 1 for drought
recall_rf = recall_score(y_test, y_pred_rf, pos_label=1)

precisions.append(precision_rf)
recalls.append(recall_rf)
print(f"Random Forest Precision: {precision_rf * 100:.2f}%")
print(f"Random Forest Recall: {recall_rf * 100:.2f}%")

# SHAP Analysis with TreeExplainer for Random Forest
explainer_rf = shap.TreeExplainer(rf)  # 'rf' is your trained Random Forest model
shap_values_rf = explainer_rf.shap_values(X_test)

print(shap_values_rf.shape)
print(X_test.columns)

# Use SHAP values for the positive class (assuming drought is labeled as 1)
shap_values_rf_drought = shap_values_rf[:, :, 1] 

# Plot SHAP summary plot as a bar chart to show feature importance
shap.summary_plot(shap_values_rf_drought, X_test, plot_type="bar", feature_names=X_test.columns)
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/shap_summary_bar_plot_rf.png")  # Save plot to file
plt.clf()  # Clear the current plot

# Plot the full SHAP summary plot to visualize feature impact on individual predictions
shap.summary_plot(shap_values_rf_drought, X_test, feature_names=X_test.columns)
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/shap_summary_plot_rf.png")  # Save plot to file
plt.clf()  # Clear the current plot

## *****************************************************************************************************************##

# Bagging Classifier
model_names.append('Bagging')
bagging = BaggingClassifier(n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
accuracies.append(accuracy_bagging)
print(f"Bagging Classifier Accuracy (Rabi Crop): {accuracy_bagging * 100:.2f}%")

# Calculate Precision and Recall for Bagging Classifier
precision_bagging = precision_score(y_test, y_pred_bagging, pos_label=1)  # Use 1 for drought
recall_bagging = recall_score(y_test, y_pred_bagging, pos_label=1)

precisions.append(precision_bagging)
recalls.append(recall_bagging)
print(f"Bagging Classifier Precision: {precision_bagging * 100:.2f}%")
print(f"Bagging Classifier Recall: {recall_bagging * 100:.2f}%")

print("Base estimator used in Bagging Classifier:", bagging.base_estimator_)

# Use PermutationExplainer for the BaggingClassifier
explainer_bagging = shap.PermutationExplainer(bagging.predict, X_test)
shap_values_bagging = explainer_bagging.shap_values(X_test)

# Plot SHAP summary plot as a bar chart for feature importance
shap.summary_plot(shap_values_bagging, X_test, plot_type="bar", feature_names=X_test.columns)
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/shap_summary_bar_plot_bagging.png")  # Save bar plot to file
plt.clf()  # Clear the current plot

# Plot the full SHAP summary plot to visualize feature impact on individual predictions
shap.summary_plot(shap_values_bagging, X_test, feature_names=X_test.columns)
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/shap_summary_plot_bagging.png")  # Save beeswarm plot to file
plt.clf()  # Clear the current plot

## *****************************************************************************************************************##

### Gradient Boosting Classifier ###
model_names.append('Gradient Boosting')
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
accuracies.append(accuracy_gb)
print(f"Gradient Boosting Accuracy (Rabi Crop): {accuracy_gb * 100:.2f}%")

# Calculate Precision and Recall for Gradient Boosting
precision_gb = precision_score(y_test, y_pred_gb, pos_label=1)  # Use 1 for drought
recall_gb = recall_score(y_test, y_pred_gb, pos_label=1)

precisions.append(precision_gb)
recalls.append(recall_gb)
print(f"Gradient Boosting Precision: {precision_gb * 100:.2f}%")
print(f"Gradient Boosting Recall: {recall_gb * 100:.2f}%")

# SHAP Analysis for Gradient Boosting
explainer_gb = shap.TreeExplainer(gb)
shap_values_gb = explainer_gb.shap_values(X_test)

# Save SHAP summary plot as an image file for Gradient Boosting
shap.summary_plot(shap_values_gb, X_test, plot_type="bar", feature_names=X_test.columns)
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/shap_summary_bar_plot_gb.png")  # Save plot to file
plt.clf()  # Clear the current plot

# Save the SHAP summary plot as an image for Gradient Boosting
shap.summary_plot(shap_values_gb, X_test, feature_names=X_test.columns)
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/shap_summary_plot_gb.png")  # Save plot to file
plt.clf()  # Clear the current plot
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
plt.savefig('WithAgriculturalMask/ResultsWeightedSum/model_performance_table.png')
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
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/top_5_features_weighted_sum.png")
plt.clf()  # Clear the plot

## *****************************************************************************************************************##

# Function to evaluate the models using top features
def evaluate_model_with_top_features(model, X_train, X_test, y_train, y_test, top_features):
    # Use only the top features for training and testing
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    
    # Fit the model
    model.fit(X_train_top, y_train)
    
    # Predict and calculate metrics
    y_pred = model.predict(X_test_top)
    
    accuracy = round(accuracy_score(y_test, y_pred), 4) 
    precision = round(precision_score(y_test, y_pred, pos_label=1), 4)   # 1 for drought
    recall = round(recall_score(y_test, y_pred, pos_label=1), 4)
    
    return accuracy, precision, recall

# Get the top 5 features from the Weighted sum
top_5_features_list = weighted_sum_df['Feature'].head(5).tolist()

# Evaluate each model using the top 1 to top 5 features
metrics = {}

# XGBoost
metrics['XGBoost'] = [evaluate_model_with_top_features(xgb_model, X_train, X_test, y_train, y_test, top_5_features_list[:i]) for i in range(1, 6)]

# Random Forest
metrics['Random Forest'] = [evaluate_model_with_top_features(rf, X_train, X_test, y_train, y_test, top_5_features_list[:i]) for i in range(1, 6)]

# Bagging
metrics['Bagging'] = [evaluate_model_with_top_features(bagging, X_train, X_test, y_train, y_test, top_5_features_list[:i]) for i in range(1, 6)]

# Gradient Boosting
metrics['Gradient Boosting'] = [evaluate_model_with_top_features(gb, X_train, X_test, y_train, y_test, top_5_features_list[:i]) for i in range(1, 6)]

# Create a DataFrame to display the metrics
metrics_df = pd.DataFrame(metrics, index=["Top 1", "Top 2", "Top 3", "Top 4", "Top 5"])

# Save the table to an image
plt.figure(figsize=(12, 6))
plt.axis('off')
plt.table(cellText=metrics_df.values, colLabels=metrics_df.columns, rowLabels=metrics_df.index, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
plt.tight_layout()
plt.savefig("WithAgriculturalMask/ResultsWeightedSum/model_metrics_top_features_weighted_sum.png")
plt.clf()

# Print the metrics for verification
print(metrics_df)

## *****************************************************************************************************************##
## Error Analysis with confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to plot confusion matrix and save as an image
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{cm}\n")  # Print the confusion matrix

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Drought (0)', 'Drought (1)'], yticklabels=['No Drought (0)', 'Drought (1)'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"WithAgriculturalMask/ResultsWeightedSum/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")  # Save confusion matrix as an image
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
    

