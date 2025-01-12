import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def create_lag_features(df, target_col, lag=1, rolling_windows=None):
    """Adds lag, rolling, and difference features for the target column."""
    # Generate lag features
    for i in range(1, lag + 1):
        df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)

    # Generate rolling statistics if specified
    if rolling_windows:
        for window in rolling_windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_sum_{window}'] = df[target_col].rolling(window=window).sum()

    # Generate difference features
    for i in range(1, lag + 1):
        df[f'{target_col}_diff_{i}'] = df[target_col].diff(periods=i)

    return df

def load():
    """Load the model and its feature names."""
    with open("xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("feature_names.pkl", "rb") as file:
        feature_names = pickle.load(file)
    print("Model and feature names loaded successfully.")
    return model, feature_names

def align_features(df, feature_names):
    """Ensure the DataFrame has the correct feature columns."""
    # Add missing columns with default value 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    # Drop extra columns
    df = df[feature_names]
    return df

def check(df):
    """Evaluate the model's predictions against the actual data."""
    # Load model and feature names
    model, feature_names = load()

    # Separate features and target variable
    X = df.drop(['Symbol','Total Revenue'], axis=1)
    y_test = df['Total Revenue']

    # Align features with the model's expectations
    X = align_features(X, feature_names)

    # Predict using the model
    y_pred = model.predict(X)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    normalized_mse = mse / np.var(y_test)
    r2 = r2_score(y_test, y_pred)
    # חישוב ממוצע הערכים האמיתיים
    mean_actual = np.mean(np.abs(y_test))

    # חישוב סטיית התקן של השגיאות
    errors = y_test - y_pred
    std_deviation = np.std(errors)

    # נירמול סטיית התקן
    normalized_std = std_deviation / mean_actual

    print(f"סטיית התקן: {std_deviation}")
    print(f"סטיית התקן המנורמלת: {normalized_std}")
    # חישוב ההבדל באחוזים בין הערכים האמיתיים ל-Total Revenue ב-X
    # חישוב אחוז ההבדל בין הערכים האמיתיים ל-Total Revenue
    comparison_results = compare_sales_changes(X, y_test, y_pred)

    print(f'Normalized MSE: {normalized_mse:.4f}')
    print(f'R² Score: {r2 * 100:.2f}%')

    importance = model.get_booster().get_score(importance_type='weight')  # You can use 'weight', 'gain', or 'cover'

    # Convert the dictionary to a sorted list of tuples for plotting
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Plot the feature importances
    features, scores = zip(*importance)
    plt.barh(features, scores)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()
    # Add comparison columns to the DataFrame for analysis
    # חישוב תחילה של ההכנסה החזויה, הטעות והטעות באחוזים
    df['Predicted Revenue'] = y_pred
    df['Prediction Error'] = y_pred - y_test
    df['Percentage Error'] = (df['Prediction Error'] / y_test) * 100

    # מחיקת שורות עם ערכים inf או NaN בתוצאה החזויה
    df_cleaned = df[~df['Percentage Error'].isin([float('inf'), float('-inf')])]
    df_cleaned = df_cleaned.dropna(subset=['Percentage Error'])

    # הצגת הנתונים אחרי ניקוי
    print("\nSample Results:")
    print(df_cleaned[[ 'Total Revenue', 'Predicted Revenue', 'Prediction Error', 'Percentage Error']].head(10))

    # חישוב אחוז ההצלחה: טעות פחות מ-20%
    success_rate = (df_cleaned['Percentage Error'] <= 20).mean()
    print(df_cleaned.describe())
    # הצגת אחוז ההצלחה
    print(f"Success rate: {success_rate * 100:.2f}%")

    return df
def scale_features(X_train, X_test):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def clean_dataframe(df):
    """Cleans the dataset by handling NaNs and infinities."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def prepare_Data(file_path, excel_file, output_file):
    """Prepares the data for prediction and evaluation."""
    # Read and process data
    df = pd.read_csv(file_path)
    df_excel = pd.read_excel(excel_file)

    # Filter and merge data
    df_2022 = df[df['Year'] == 2022].drop(['Year'], axis=1)
    df_2022.rename(columns={'Total Revenue': 'Total Revenue_2022'}, inplace=True)
    df_2023 = df_excel[df_excel['Year'] == 2023][['Symbol', 'Total Revenue']]
    df_combined = pd.merge(df_2022, df_2023, on='Symbol', how='left', suffixes=('', '_2022'))

    # Clean and preprocess data
    df_combined = clean_dataframe(df_combined)

    # Add lag for Total Revenue (last year's revenue as a predictor)
    df_combined['Total Revenue_lag_1'] = df_combined['Total Revenue']
    df_combined.rename(columns={'Total Revenue_2022': 'Total Revenue'}, inplace=True)

    # Add other lag and engineered features
    df_combined = create_lag_features(df_combined, target_col='Total Revenue', lag=3, rolling_windows=[3, 6])
    df_combined.dropna(inplace=True)

    # Predict and evaluate
    df_predicted = check(df_combined)

    # Plot results
   # plot_percentage_gap(df_predicted)



def compare_sales_changes(X, y_test, y_pred):
    """
    Compares the percentage change in sales between years for actual and predicted values.

    Parameters:
        X (DataFrame): Original dataset with columns including 'Total Revenue' for year-based comparison.
        y_test (array-like): Actual sales values for the test set.
        y_pred (array-like): Predicted sales values for the test set.

    Returns:
        DataFrame: Summary of percentage change comparisons.
    """
#    print(y_test- X['Total Revenue'])
    # אחוז השינוי במכירות בפועל לפי השנים
    return None


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
prepare_Data(file_path, excel_file, output_file)
