# AOMLProj

Bankruptcy Prediction using the New Connect Dataset

Problem Statement

Corporate bankruptcy and default prediction is a critical task in financial risk assessment. The ability to accurately predict a company's likelihood of default enables financial institutions, investors, and stakeholders to make informed decisions. This project aims to build a machine learning model to predict corporate default based on various financial ratios and indicators.

Dataset Source and Description

Dataset: New Connect Market - Corporate Default Prediction

Description: This dataset contains financial data of companies from the New Connect market, including various financial ratios, balance sheet components, and profitability indicators. The target variable indicates whether a company defaulted or not.

Features:

Financial ratios such as Return on Assets, Net Profit Margin, and Debt Ratios.

Cash flow indicators.

Time-based factors such as year and operating cycle details.

Categorical and numerical attributes related to company financial health.

#Solution Details

Model Selection

The project employs XGBoost (Extreme Gradient Boosting) as the primary model for predicting corporate default. XGBoost is chosen for its high performance in classification tasks, ability to handle imbalanced data, and feature importance extraction.

#Implementation Steps

Data Preprocessing

Load the dataset and handle missing values.

Encode categorical features.

Split into training and test sets.

Address class imbalance using resampling techniques.

Feature Selection using XGBoost

Train an initial XGBoost model on the original dataset.

Extract feature importance scores.

Select the top features based on a predefined threshold.

Model Training and Evaluation

Train a refined XGBoost model using only the selected important features.

Evaluate the model using Accuracy and AUC-ROC score.

#Code Implementation

import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

# Train initial model
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Get feature importance
xgb_importances = xgb_model.feature_importances_
xgb_feature_importance_dict = dict(zip(X_train.columns, xgb_importances))

# Sort and select top features
sorted_xgb_features = sorted(xgb_feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
threshold = 0.03
important_xgb_features = [feature for feature, importance in sorted_xgb_features if importance >= threshold]

# Train new model with selected features
X_train_selected_xgb = X_train_resampled[important_xgb_features]
X_test_selected_xgb = X_test[important_xgb_features]
xgb_model.fit(X_train_selected_xgb, y_train_resampled)

# Evaluate Model
y_pred_selected_xgb = xgb_model.predict(X_test_selected_xgb)
new_accuracy_xgb = accuracy_score(y_test, y_pred_selected_xgb)
new_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test_selected_xgb)[:, 1])

print(f"\nNew Model Accuracy (XGBoost): {new_accuracy_xgb:.4f}")
print(f"New Model AUC-ROC (XGBoost): {new_auc_xgb:.4f}")

#Key Features Selected

Foreign Service / Net Profit

Return on Assets

Operating Expenses / Current Liabilities

Assets Ratio

Net Profit / Current Liabilities

Rate Debt Security

Cash Flow from Investing Activities / Equity Shareholders of the Parent

Trade Payables / Equity Shareholders of the Parent

Cost of Sales + General and Administrative Costs + Operating Expenses) / Profit

#Results and Findings

Model Performance: The refined XGBoost model with selected features showed improved efficiency.

AUC-ROC Score: Measures the model's ability to distinguish between defaulting and non-defaulting companies.

Top Features: The most influential financial ratios that contribute to corporate default prediction.

#Future Work

Experiment with other models like Random Forest and Neural Networks.

Incorporate additional financial indicators for better predictive power.

Tune hyperparameters to optimize performance further.

Conclusion

This project successfully develops an XGBoost-based corporate default prediction model using financial ratios. Feature selection improves model interpretability while maintaining high accuracy and robustness. The results demonstrate the effectiveness of machine learning in financial risk assessment.
