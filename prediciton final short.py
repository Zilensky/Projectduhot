import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import numpy as np

# Load the dataset
file_path = r'C:\Users\Ascending\Desktop\combined_financial_data_all_stocks.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Filter rows for years < 2023 for training
df_train = df[df['Year'] < 2023]

# List of target columns to predict
target_columns = ['Net Debt', 'Total Debt', 'Tangible Book Value',
                  'Invested Capital', 'Working Capital', 'Net Tangible Assets']

# Clean the target columns: remove rows with NaN, infinity, or negative values
for col in target_columns:
    df_train = df_train[df_train[col].notna()]  # Remove NaNs
    df_train = df_train[df_train[col] >= 0]  # Remove negative values
    df_train = df_train[np.isfinite(df_train[col])]  # Remove infinite values

# Apply log transformation to reduce skewness for all target columns
for col in target_columns:
    df_train[col] = np.log1p(df_train[col])  # log1p is log(1 + x) to handle zeroes

# Define feature columns
features = ['Year', 'Ordinary Shares Number', 'Share Issued']

# Impute missing values in feature columns with the median
imputer = SimpleImputer(strategy='median')
X = df_train[features]
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data for each target and train a model for each
predictions_2023 = {}
df_2023 = df[df['Year'] == 2023]
X_2023 = df_2023[features]
X_2023_imputed = imputer.transform(X_2023)
X_2023_scaled = scaler.transform(X_2023_imputed)

# Use the best hyperparameters for all targets
best_params = {
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 0.9
}

for target in target_columns:
    y = df_train[target]

    # Train-test split for each target
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # XGBoost model with fixed hyperparameters
    best_xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                      learning_rate=best_params['learning_rate'],
                                      max_depth=best_params['max_depth'],
                                      n_estimators=best_params['n_estimators'],
                                      subsample=best_params['subsample'],
                                      random_state=42)

    # Train the model
    best_xgb_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = best_xgb_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for {target}: {mse}')

    # Predict for 2023
    y_2023_pred = best_xgb_model.predict(X_2023_scaled)

    # Reverse the log transformation for predictions
    y_2023_pred = np.expm1(y_2023_pred)  # Reverse log1p to original scale

    # Store the predictions for 2023
    predictions_2023[target] = y_2023_pred

# Assign predictions for all targets back to the DataFrame
df_2023 = df_2023.copy()  # Avoid SettingWithCopyWarning
for target in target_columns:
    df_2023[f'Predicted {target}'] = predictions_2023[target]

# Save the df_2023 with predictions to an Excel file
output_file_path = r'C:\Users\Ascending\Desktop\results.xlsx'
df_2023[['Symbol', 'Year'] + [f'Predicted {col}' for col in target_columns]].to_excel(output_file_path, index=False)

print(f"Predictions saved successfully to {output_file_path}")
