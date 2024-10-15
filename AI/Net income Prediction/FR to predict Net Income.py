import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score

def clean_dataframe(df):
    # Replace inf and -inf with large and small finite numbers respectively
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaNs with zeros
    df.fillna(0, inplace=True)
    return df


def filter_outliers(df):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])

    # Calculate Q1, Q3, and IQR for outlier filtering
    Q1 = numeric_cols.quantile(0.05)
    Q3 = numeric_cols.quantile(0.8)
    IQR = Q3 - Q1

    # Filter out outliers in the numeric columns
    df_filtered = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_filtered
def predict_net_income(df):

    # Separate features and target variable
    X = df.drop(['Symbol', 'Net Income'], axis=1)
    y = df['Net Income']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the regression model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2 * 100:.2f}%')
    within_5_percent = np.abs((y_pred - y_test) / y_test) <= 0.05
    accuracy_within_5_percent = np.mean(within_5_percent) * 100
    print(f'Accuracy within 5%: {accuracy_within_5_percent:.2f}%')
    # Predict net income for the entire dataset
    df['Predicted Net Income'] = model.predict(X)

    return df

def print_diffrences(df_predicted):
    df_predicted['Difference'] = abs(df_predicted['Net Income'] - df_predicted['Predicted Net Income'])

    # Get the top 10 rows with the largest difference
    top_10_gap = df_predicted[['Symbol', 'Net Income', 'Predicted Net Income', 'Difference']].nlargest(10, 'Difference')

    # Print the top 10
    print("Top 10 symbols with the largest gap between predicted and real Net Income:")
    print(top_10_gap)


def combine_and_predict(file_path, excel_file, output_file):
    # Read and process data
    df = pd.read_csv(file_path)
    df_excel = pd.read_excel(excel_file)

    # Filter and merge data
    df_2022 = df[df['Year'] == 2021].drop(['Year'], axis=1)
    df_2023 = df_excel[df_excel['Year'] == 2022][['Symbol', 'Net Income']]
    df_combined = pd.merge(df_2022, df_2023, on='Symbol', how='left', suffixes=('', '_2022'))

    # Clean the data
    df_combined = clean_dataframe(df_combined)
    df_combined=filter_outliers(df_combined)
    # Predict Net Income
    df_predicted = predict_net_income(df_combined)
    plot_percentage_gap(df_predicted)
    # Save the DataFrame with predictions
    #df_predicted.to_csv(output_file, index=False)

def plot_percentage_gap(df_predicted):
    # Calculate the percentage change between predicted and actual net income
    df_predicted['Percentage Gap'] = ((df_predicted['Predicted Net Income'] - df_predicted['Net Income']) / df_predicted['Net Income']) * 100

    # Sort by symbol for better visualization
    df_sorted = df_predicted.sort_values(by='Symbol')

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the percentage gap
    plt.bar(df_sorted['Symbol'], df_sorted['Percentage Gap'], color='blue')

    # Add labels and title
    plt.xlabel('Company Symbols')
    plt.ylabel('Percentage Gap (%)')
    plt.title('Percentage Gap Between Predicted and Actual Net Income by Company')
    plt.xticks(rotation=90, fontsize=8)  # Rotate company symbols for better readability

    # Show the plot
    plt.tight_layout()
    plt.show()
# Usage
file_path = 'financial_comparisons.csv'  # Your file with 2022 financial data
excel_file = 'combined_financial_data_all_stocks.xlsx'  # Your file with 2023 data
output_file = 'financial_predictions_with_regression.csv'  # Output file
combine_and_predict(file_path, excel_file, output_file)
