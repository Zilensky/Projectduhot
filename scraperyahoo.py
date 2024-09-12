import yfinance as yf
import pandas as pd
import openpyxl

# הגדר את הסימבול של החברה
symbol = 'AURA.TA'  # לדוגמה, Apple

# הורדת הנתונים הכספיים
stock = yf.Ticker(symbol)

# דוחות כספיים היסטוריים - מאזן
balance_sheet = stock.balance_sheet

# דוחות רווח והפסד היסטוריים
income_statement = stock.financials

# דוחות תזרים מזומנים היסטוריים
cash_flow = stock.cashflow

# בדיקה אם קיים דוח של שינויים בהון עצמי
try:
    changes_in_equity = stock.get_shares()
except Exception as e:
    changes_in_equity = None
    print("לא נמצאו נתונים על שינויים בהון עצמי.")

# שמירה לקובץ Excel
with pd.ExcelWriter('financial_reports.xlsx') as writer:
    balance_sheet.to_excel(writer, sheet_name='Balance Sheet')
    income_statement.to_excel(writer, sheet_name='Income Statement')
    cash_flow.to_excel(writer, sheet_name='Cash Flow')
    if changes_in_equity is not None:
        changes_in_equity.to_excel(writer, sheet_name='Changes in Equity')
