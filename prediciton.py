import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Drop rows with missing values in the target column (Net Debt)
df_train = df_train.dropna(subset=['Net Debt'])

# Apply log transformation to the target (Net Debt) to reduce skewness
df_train['Net Debt'] = np.log1p(df_train['Net Debt'])  # log1p is log(1 + x) to handle zeroes

# Define features and target variable
features = ['Year', 'Ordinary Shares Number', 'Share Issued', 'Total Debt',
            'Tangible Book Value', 'Invested Capital', 'Working Capital']
target = 'Net Debt'

X = df_train[features]
y = df_train[target]

# Impute missing values in feature columns with the median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost model with cross-validation
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Set up hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0]
}

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_xgb_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Best Parameters: {grid_search.best_params_}')

# Predict for the year 2023
df_2023 = df[df['Year'] == 2023]
X_2023 = df_2023[features]

# Impute missing values and scale features for 2023 data
X_2023_imputed = imputer.transform(X_2023)
X_2023_scaled = scaler.transform(X_2023_imputed)

# Predict for 2023
y_2023_pred = best_xgb_model.predict(X_2023_scaled)

# Reverse the log transformation for predictions
y_2023_pred = np.expm1(y_2023_pred)  # Reverse log1p to original scale

# Assign predictions back to the DataFrame
df_2023 = df_2023.copy()  # Avoid SettingWithCopyWarning
df_2023['Predicted Net Debt'] = y_2023_pred

# Show predictions for 2023
print(df_2023[['Symbol', 'Year', 'Predicted Net Debt']])
