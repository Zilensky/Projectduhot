import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def combine_data_from_same_file(file_path):
    # Load the data
    df = pd.read_excel(file_path)

    # Filter data for 2022
    df_2022 = df[df['Year'] == 2022]

    # Filter data for 2023 and keep only 'Symbol' and 'Net Income'
    df_2023 = df[df['Year'] == 2023][['Symbol', 'Net Income']]

    # Merge 2022 data with 2023 net income
    df_combined = pd.merge(df_2022, df_2023, on='Symbol', how='left', suffixes=('_2022', '_2023'))

    return df_combined


def preprocess_data(df):
    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Drop rows where Net Income for 2023 is 0 (if these rows have no relevant data)
    df = df[df['Net Income_2023'] != 0]

    return df


def train_linear_regression_model(df):
    # Prepare feature and target variables
    X = df.drop(columns=['Symbol', 'Year', 'Net Income_2023'])
    y = df['Net Income_2023']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X_test, y_test, y_pred


def calculate_accuracy_within_std(y_test, y_pred, std_threshold=0.1):
    # Calculate the standard deviation of the actual Net Income
    std_actual = np.std(y_test)

    # Calculate the differences
    differences = np.abs(y_pred - y_test)

    # Calculate percentage differences
    percentage_differences = differences / std_actual

    # Check how many predictions are within the standard deviation threshold
    accurate_predictions = np.sum(percentage_differences <= std_threshold)
    total_predictions = len(y_test)

    accuracy = accurate_predictions / total_predictions * 100

    return percentage_differences, accuracy


def plot_percentage_differences(y_test, y_pred):
    # Calculate percentage differences
    percentage_differences = ((y_pred - y_test) / y_test) * 100

    # Create a DataFrame for plotting
    df_plot = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Percentage Difference': percentage_differences})

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(df_plot.index, df_plot['Percentage Difference'], color='skyblue')
    plt.xlabel('Index')
    plt.ylabel('Percentage Difference (%)')
    plt.title('Percentage Difference between Actual and Predicted Net Income for 2023')
    plt.axhline(0, color='gray', linestyle='--')
    plt.show()


def plot_distribution_of_percentage_differences(percentage_differences, std_threshold):
    # Plot histogram of percentage differences
    plt.figure(figsize=(12, 6))
    plt.hist(percentage_differences * 100, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(std_threshold * 100, color='red', linestyle='--', label=f'STD Threshold ({std_threshold * 100}%)')
    plt.xlabel('Percentage Difference (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Percentage Differences')
    plt.legend()
    plt.show()


# Usage
file_path = 'combined_financial_data_all_stocks.xlsx'  # Your file with data
df_combined = combine_data_from_same_file(file_path)
df_preprocessed = preprocess_data(df_combined)
model, mse, r2, X_test, y_test, y_pred = train_linear_regression_model(df_preprocessed)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Calculate and print accuracy within std deviation threshold
percentage_differences, accuracy = calculate_accuracy_within_std(y_test, y_pred, std_threshold=1)
print(f"Accuracy within ±0.1 standard deviations: {accuracy:.2f}%")

# Plot percentage differences
plot_percentage_differences(y_test, y_pred)

# Plot distribution of percentage differences
plot_distribution_of_percentage_differences(percentage_differences, std_threshold=0)
