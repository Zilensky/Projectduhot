# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PqLFTCvZn8ur4r_USveYIaM_5DtGju_m
"""

# prompt: add the documentand conver t it to df /content/combined_financial_data_all_stocks (1).xlsx

import pandas as pd

# Read the Excel file into a pandas DataFrame
df = pd.read_excel('../Files/combined_financial_data_all_stocks.xlsx')

# Print some info about the DataFrame
print(df.head())
print(df.info())

df

# prompt: print all the coloums

for i in df.columns:
  print(i)

# prompt: create a list with all the colums without the variable "Net Income"

columns_without_net_income = [col for col in df.columns if col != 'Net Income' and col !='Symbol' and col!='Year']

# prompt: clean all the data: null=0

df[columns_without_net_income] = df[columns_without_net_income].fillna(0)
df['Net Income'] = df['Net Income'].fillna(0)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# קריאת הנתונים לקובץ DataFrame


# הגדרת משתני התכנים והתגיות
X = df[columns_without_net_income]  # כלל את כל העמודות מלבד Net Income
y = df['Net Income']  # העמודה שמייצגת את הרווחים

# חלוקת הנתונים לסטים של אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# יצירת המודל
model = LinearRegression()
model.fit(X_train, y_train)

# חיזוי באמצעות המודל
y_pred = model.predict(X_test)

# הערכת הביצועים של המודל
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# הדפסת מקדמי המודל
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# prompt: print the regression function

print(f"Regression Function: Net Income = {model.intercept_} + {model.coef_[0]} * Total Revenue + {model.coef_[1]} * Cost Of Revenue + {model.coef_[2]} * Operating Expense + {model.coef_[3]} * Interest Expense + {model.coef_[4]} * Net Debt + {model.coef_[5]} * Invested Capital + {model.coef_[6]} * Total Assets + {model.coef_[7]} * Net PPE + {model.coef_[8]} * EBITDA + {model.coef_[9]} * Gross Profit")

# prompt: COMPUTE THE ACCURECY OF THE MODEL IN PERECENT

from sklearn.metrics import accuracy_score
# Assuming your model is a classification model
# If your model is a regression model, accuracy is not the right metric.
# Consider using R-squared or mean squared error instead.

# If you have a classification problem, you need to convert your predictions to classes.
# For example, if you're predicting a binary outcome (0 or 1), you might use a threshold:
y_pred_classes = [1 if pred > 0.5 else 0 for pred in y_pred]

# Calculate accuracy


# prompt: now create a AI that get as traning samples all the data for each company over the years 2019-2022 and predict all the variables vaues in 2023. predict all the colums in 2023

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the Excel file into a pandas DataFrame


# Filter data for years 2019-2022
df_train = df[df['Year'].isin([2019, 2020, 2021])]

# Separate features (X) and target variable (y)
columns_without_symbol_year = [col for col in df.columns if col not in ['Symbol', 'Year']]
X_train = df_train[columns_without_symbol_year].drop('Net Income', axis=1)
y_train = df_train['Net Income']

# Filter data for 2023
df_test = df[df['Year'] == 2022]
X_test = df_test[columns_without_symbol_year].drop('Net Income', axis=1)

# Fill missing values with 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict all columns for 2023
y_pred = model.predict(X_test)

# Create a DataFrame for predictions
predictions_df = pd.DataFrame(y_pred, columns=['Net Income'])

# Add other columns with predicted values (assuming you want to predict all columns)
# You might need to adjust this based on your specific needs
for col in columns_without_symbol_year:
  if col != 'Net Income':
    # Predict the values for the current column
    y_pred_col = model.predict(X_test)
    # Add the predictions to the DataFrame
    predictions_df[col] = y_pred_col

# Add Symbol and Year columns to the predictions DataFrame
predictions_df['Symbol'] = df_test['Symbol']
predictions_df['Year'] = df_test['Year']

# Print the predictions
print(predictions_df)

# prompt: calculate MSE and R^2 Score in precentfor every coulum

# Calculate MSE and R^2 for each column
for col in columns_without_symbol_year:
  if col != 'Net Income':
    y_test_col = df_test[col]
    y_pred_col = predictions_df[col]
    mse = mean_squared_error(y_test_col, y_pred_col)
    r2 = r2_score(y_test_col, y_pred_col)
    print(f"Column: {col}")
    print(f"  Mean Squared Error: {mse}")
    print(f"  R^2 Score: {r2 * 100:.2f}%")


# Feature importance (Optional)



df.fillna(0, inplace=True)

# Calculating financial ratios for each company

# Liquidity Ratios
df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']
df['Quick Ratio'] = (df['Current Assets'] - df['Inventory']) / df['Current Liabilities']

# Profitability Ratios
df['Net Profit Margin'] = df['Net Income'] / df['Total Revenue']
df['Return on Assets (ROA)'] = df['Net Income'] / df['Total Assets']
df['Return on Equity (ROE)'] = df['Net Income'] / df['Common Stock Equity']

# Leverage Ratios
df['Debt-to-Equity Ratio'] = df['Total Liabilities Net Minority Interest'] / df['Common Stock Equity']
df['Debt Ratio'] = df['Total Liabilities Net Minority Interest'] / df['Total Assets']

# Efficiency Ratios
df['Asset Turnover Ratio'] = df['Total Revenue'] / df['Total Assets']
df['Inventory Turnover Ratio'] = df['Cost Of Revenue'] / df['Inventory']

# Market Ratios
df['EPS'] = (df['Net Income'] - df['Preferred Stock Dividends']) / df['Diluted Average Shares']
# Assuming Market Price per Share is not available; if available, calculate P/E Ratio
# df['P/E Ratio'] = df['Market Price per Share'] / df['EPS']

# Display the dataframe with computed ratios
print(df[['Symbol', 'Year', 'Current Ratio', 'Quick Ratio', 'Net Profit Margin',
          'Return on Assets (ROA)', 'Return on Equity (ROE)', 'Debt-to-Equity Ratio',
          'Debt Ratio', 'Asset Turnover Ratio', 'Inventory Turnover Ratio', 'EPS']])

# prompt: print the function

print(f"Regression Function: Net Income = {model.intercept_} + {model.coef_[0]} * Total Revenue + {model.coef_[1]} * Cost Of Revenue + {model.coef_[2]} * Operating Expense + {model.coef_[3]} * Interest Expense + {model.coef_[4]} * Net Debt + {model.coef_[5]} * Invested Capital + {model.coef_[6]} * Total Assets + {model.coef_[7]} * Net PPE + {model.coef_[8]} * EBITDA + {model.coef_[9]} * Gross Profit")

import pandas as pd

# Assuming you already have your calculated financial ratios in the 'df' DataFrame
# Select only the columns you need for the financial comparisons
df_comparisons = df[['Symbol', 'Year', 'Current Ratio', 'Quick Ratio', 'Net Profit Margin',
                     'Return on Assets (ROA)', 'Return on Equity (ROE)', 'Debt-to-Equity Ratio',
                     'Debt Ratio', 'Asset Turnover Ratio', 'Inventory Turnover Ratio', 'EPS']]

# Convert the new DataFrame to a CSV file
df_comparisons.to_csv('financial_comparisons.csv', index=False)

print("CSV file created successfully.")
