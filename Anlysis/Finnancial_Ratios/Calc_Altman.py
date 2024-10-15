def CalcAltman(symbol):
    import yfinance as yf
    import pandas as pd
    from colorama import Style, Fore

    # Define the symbol for the company
      # Example: Aura

    # Download financial data
    stock = yf.Ticker(symbol)

    # Historical financial statements
    balance_sheet = stock.balance_sheet

    income_statement = stock.financials

    cash_flow = stock.cashflow

    # Extract relevant values from the balance sheet
    working_capital = balance_sheet.loc['Working Capital'].values[0]

    retained_earnings = balance_sheet.loc['Retained Earnings'].values[0]

    total_assets = balance_sheet.loc['Total Assets'].values[0]

    total_liabilities = balance_sheet.loc['Stockholders Equity'].values[0]
    total_liabilities = total_assets - total_liabilities

    # Extract EBIT from income statement
    ebit = income_statement.loc['EBIT'].values[0]
    print(ebit)
    # Get the market value of equity
    market_value_of_equity = stock.info['marketCap']

    # Get total sales (revenue)
    sales = income_statement.loc['Total Revenue'].values[0]

    # Calculate Z-Score components
    X1 = working_capital / total_assets
    X2 = retained_earnings / total_assets
    X3 = ebit / total_assets
    X4 = market_value_of_equity / total_liabilities
    X5 = sales / total_assets

    # Calculate Z-Score
    Z_score = (1.2 * X1) + (1.4 * X2) + (3.3 * X3) + (0.6 * X4) + X5

    # Display the Z-Score
    print(f"Altman Z-Score for {symbol}: {Z_score}")

    # Determine the financial health of the company
    if Z_score > 2:
        status = Fore.GREEN + "Good" + Style.RESET_ALL  # Green for good condition
    elif 1 <= Z_score <= 2:
        status = Fore.YELLOW + "Moderate" + Style.RESET_ALL  # Yellow for moderate condition
    else:
        status = Fore.RED + "Poor" + Style.RESET_ALL  # Red for poor condition

    print(f"Financial Status: {status}")
    return Z_score,status

if __name__ == '__main__':
    symbol = 'IBI.TA'
    CalcAltman(symbol)