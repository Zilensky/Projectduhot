import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def clean_dataframe(df):
    """Cleans the dataset by handling NaNs and infinities."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def filter_outliers(df):
    """Removes outliers based on IQR."""
    numeric_cols = df.select_dtypes(include=[np.number])
    Q1 = numeric_cols.quantile(0.1)
    Q3 = numeric_cols.quantile(0.8)
    IQR = Q3 - Q1
    return df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

def create_lagged_features(df, target_column, lags=3):
    """Creates lagged features for the target column."""
    for lag in range(1, lags + 1):
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df

def add_rolling_features(df, target_column, windows=[3, 6]):
    """Adds rolling mean and rolling sum features."""
    for window in windows:
        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
        df[f'{target_column}_rolling_sum_{window}'] = df[target_column].rolling(window=window).sum()
    return df

def add_difference_features(df, target_column):
    """Adds differencing features to capture trends."""
    df[f'{target_column}_diff_1'] = df[target_column].diff(1)
    df[f'{target_column}_diff_2'] = df[target_column].diff(2)
    return df

def prepare_features(df, target_column):
    """Prepares the dataset by adding memory-based features."""
    df = create_lagged_features(df, target_column)
    df = add_rolling_features(df, target_column)
    df = add_difference_features(df, target_column)
    return df

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def create_lag_features(df, target_col, lag=1):
    """Adds lag features for the target column."""
    for i in range(1, lag + 1):
        df[f'{target_col}_lag{i}'] = df[target_col].shift(i)
    return df

def scale_features(X_train, X_test):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def tune_xgboost_hyperparameters(X_train, y_train):
    """Tunes the XGBoost hyperparameters using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    xgb = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def predict_net_income(dfPast, tune_model=False, lag_features=1,model=None):
    lag_features = 1
    """Trains the XGBoost model with optional tuning and lag features."""
    df = clean_dataframe(dfPast)
    df = create_lag_features(df, target_col='Total Revenue', lag=lag_features)
    df.dropna(inplace=True)  # Drop rows with NaN introduced by lagging

    # Separate features and target
    X = dfPast.drop(['Symbol', 'Total Revenue'], axis=1)
    y = dfPast['Total Revenue']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    if model:
       # model.model.set_params(n_estimators=200)
        model.fit(X_train_scaled, y_train)
    else:
    # Train or tune the model
     if tune_model:
        model = tune_xgboost_hyperparameters(X_train_scaled, y_train)
     else:
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
    feature_names = X.columns.tolist()
    with open("feature_names.pkl", "wb") as file:
        pickle.dump(feature_names, file)
    with open("xgboost_model.pkl", "wb") as file:
        pickle.dump(model, file)
    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    normalized_mse = mse / np.var(y_test)
    r2 = r2_score(y_test, y_pred)

    actual_direction = np.sign(y_test.values - y_train.mean())
    predicted_direction = np.sign(y_pred - y_train.mean())
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    print(f'Normalized MSE: {normalized_mse}')
    print(f'R² Score: {r2 * 100:.2f}%')
    print(f'Directional Accuracy: {directional_accuracy:.2f}%')

    # Add predictions to the DataFrame
    df['Predicted Net Income'] = model.predict(scale_features(X, X)[0])
    return model,df
def predict_for_a_year(df,df_excel,pre,post,model):

    # Merge datasets
    df_2022 = df[df['Year'] ==pre].drop(['Year'], axis=1)
    df_2022.rename(columns={'Total Revenue': 'Total Revenue_2022'}, inplace=True)
    df_2023 = df_excel[df_excel['Year'] == post][['Symbol', 'Total Revenue']]
    df_combined = pd.merge(df_2022, df_2023, on='Symbol', how='left')

    # Add memory features
    df_combined = prepare_features(df_combined, 'Total Revenue')
    df_combined = clean_dataframe(df_combined)
    df_combined = filter_outliers(df_combined)

    # Predict and save
    return predict_net_income(df_combined,False,1,model)
def combine_and_predict(file_path, excel_file, output_file):
    """Combines data from multiple sources and predicts net income."""
    df = pd.read_csv(file_path)
    df_excel = pd.read_excel(excel_file)
    model = None
    # Merge datasets
    df_2022 = df[df['Year'] ==2021].drop(['Year'], axis=1)
    df_2022.rename(columns={'Total Revenue': 'Total Revenue_2022'}, inplace=True)
    df_2023 = df_excel[df_excel['Year'] == 2022][['Symbol', 'Total Revenue']]
    df_combined = pd.merge(df_2022, df_2023, on='Symbol', how='left')

    # Add memory features
    df_combined = prepare_features(df_combined, 'Total Revenue')
    df_combined = clean_dataframe(df_combined)
    df_combined = filter_outliers(df_combined)

    # Predict and save
    model,df_predicted = predict_net_income(df_combined,False,1,model)
    predict_for_a_year(df, df_excel, 2020, 2021,model)
   # predict_for_a_year(df, df_excel, 2019, 2020,model)

    plot_percentage_gap(df_predicted)

def plot_percentage_gap(df_predicted):
    """Plots the percentage gap between actual and predicted net income."""
    df_predicted['Percentage Gap'] = ((df_predicted['Predicted Net Income'] - df_predicted['Total Revenue']) / df_predicted['Total Revenue']) * 100
    df_sorted = df_predicted.sort_values(by='Symbol')

    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted['Symbol'], df_sorted['Percentage Gap'], color='blue')
    plt.xlabel('Company Symbols')
    plt.ylabel('Percentage Gap (%)')
    plt.title('Percentage Gap Between Predicted and Actual Net Income')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.show()

# File paths
file_path = '../Net income Prediction/financial_comparisons.csv'
excel_file = '../Net income Prediction/financial_ratios_with_altman.xlsx'
output_file = '../Net income Prediction/financial_predictions_with_memory.csv'

# Run the pipeline
combine_and_predict(file_path, excel_file, output_file)
