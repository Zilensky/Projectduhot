import pandas as pd
import yfinance

f=pd.read_excel('combined_financial_data_all_stocks.xlsx')
for i in f.columns:
    print(i)