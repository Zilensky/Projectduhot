from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt


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

   # print(f"Financial Status: {status}")
    return Z_score,status
def get_price_change(symbol, start_date, end_date):
    """Retrieve stock price data between two dates and calculate the percentage change."""
    stock = yf.Ticker(symbol)
    price_data = stock.history(start=start_date, end=end_date)
    print(price_data)
    if not price_data.empty:
        initial_price = price_data['Close'].iloc[0]
        final_price = price_data['Close'].iloc[-1]
        price_change = ((final_price - initial_price) / initial_price) * 100
        return price_change, price_data
    else:
        return None, None
def plot_histograms(dataframe):
    """Plot histograms for numerical columns in the DataFrame."""
    dataframe.hist(bins=20, figsize=(12, 8), edgecolor='black')
    plt.suptitle('Distribution of Ratios', fontsize=16)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    file_path = r'C:\Users\yairb\PycharmProjects\Projectduhot\SYMBOLS.xlsx'
    df = pd.read_excel(file_path)
    symbols = df['symb'].tolist()  # Assuming the column is named 'symb'
    demo_portfolio = {}
    # Step 3: Iterate over the symbols and calculate the ratios
    results = []
    total_money=0
    for symbol in symbols:
        try:

            SCORE,STATUS = CalcAltman( symbol)
            investment = 1
            # Step 2: Decide investment and record in portfolio
            if SCORE <2:
                investment=0
                # Step 3: Check stock price over the next 3 months

                # Step 3: Check stock price over the next 3 months
            start_date = datetime.now()
            end_date = start_date + timedelta(days=90)
            price_change, price_data = get_price_change(symbol, start_date, end_date)
            print(price_change)
            if investment==1:
                    # Calculate remaining value after 3 months
                    final_value = investment * (1 + price_change / 100)
                    demo_portfolio[symbol]["Price Change (%)"] = price_change
                    demo_portfolio[symbol]["Final Value ($)"] = final_value

                    # Print revenue and percentage change for debugging
                    revenue = final_value - investment
                    print(f"Symbol: {symbol}")
                    print(f"Revenue: {revenue:.2f}, Percentage Change: {price_change:.2f}%")

                    # Update total money
                    total_money += final_value

                    # Step 4: Plot stock price graph

                    # Step 4: Plot stock price graph

        except Exception as e:
             print(f"Error processing symbol {symbol}: {e}")


        results.append(SCORE)

    print(f"Total money left after 3 months: ${total_money:.2f}")


    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Ensure the DataFrame is not empty
    if not results_df.empty:
        print("Summary statistics for the data:")
        print(results_df.describe())  # Display summary statistics

        # Plot histograms

    else:
        print("No data to process or visualize.")