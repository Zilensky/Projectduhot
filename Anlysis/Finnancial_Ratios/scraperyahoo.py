import yfinance as yf
import pandas as pd
from colorama import Style, Fore

from Anlysis.Finnancial_Ratios.Ratios.Operations_Efficencey import calc_receivables_ratio, calc_days_sales_outstanding, \
    calc_inventory_ratio, calc_inventory_turnover_ratio, calc_inventory_days, calc_payables_days
from Anlysis.Finnancial_Ratios.Ratios.Structure import calc_leverage_ratio, calc_equity_to_assets_ratio
from Ratios.Liquidity import current_ratio,quick_ratio,calc_liquidity_ratio,calc_cashflow_to_sales_ratio
from Ratios.Earnings import calc_net_profit_margin, calc_operating_profit_margin, calc_ebitda_ratio, calc_roe, calc_roa

# Define the symbol for the company
symbol = 'AURA.TA'  # Example: Aura

# Download financial data
stock = yf.Ticker(symbol)

# Historical financial statements


def calc_stock_price(stock,symbol):
    # Download historical data for the last 5 days using the history function
    stock_data = stock.history(period="5d")

    print(symbol)
    # Print the last 5 closing prices
    closing_prices = stock_data['Close']
    print("Last 5 closing prices for:")
    print(closing_prices)

def calc_Ratios(stock,symbol):
    balance_sheet = stock.balance_sheet

    income_statement = stock.financials

    cash_flow = stock.cashflow
    print(balance_sheet)
    #יחסי נזילות
    current_ratio_value = current_ratio(balance_sheet)
    quick_ratio_value = quick_ratio(balance_sheet)
    immediate_liquidity_ratio=calc_liquidity_ratio(balance_sheet)
    cashflow_to_sales_ratio=calc_cashflow_to_sales_ratio(cash_flow,income_statement)
    # Print the
    print("יחסי נזילות:")
    print(f"{Fore.GREEN}Current Ratio (יחס שוטף) for {symbol}: {current_ratio_value:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Quick Ratio (יחס מהיר) for {symbol}: {quick_ratio_value:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Immediate Liquidity Ratio (רמת נזילות מיידית) for {symbol}: {immediate_liquidity_ratio:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Cash Flow to Sales Ratio for {symbol}: {cashflow_to_sales_ratio:.2f}{Style.RESET_ALL}")


    #יחסי רווחיות
    net_profit_margin_value = calc_net_profit_margin(income_statement)
    operating_profit_margin_value = calc_operating_profit_margin(income_statement)
    ebitda_ratio = calc_ebitda_ratio(income_statement)
    roe_value = calc_roe(balance_sheet, income_statement)
    roa_value = calc_roa(balance_sheet, income_statement)
    print("\nיחסי רווחיות")
    print(f"{Fore.GREEN}Net Profit Margin (שיעור הרווח הנקי) for {symbol}: {net_profit_margin_value:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Operating Profit Margin (שיעור הרווח התפעולי) for {symbol}: {operating_profit_margin_value:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}EBITDA to Sales Ratio for {symbol}: {ebitda_ratio:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}ROE for {symbol}: {roe_value:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}ROA for {symbol}: {roa_value:.2f}{Style.RESET_ALL}")
    #יחסי מבני הון
    market_value_of_equity = stock.info['marketCap']
    calc_leverage_ratio_rate=calc_leverage_ratio(balance_sheet,market_value_of_equity)
    calc_equity_to_assets_ratio_rate=calc_equity_to_assets_ratio(balance_sheet)
    print("\nיחסי מבני הון")
    print(f"{Fore.GREEN}Leverage ratio for {symbol}: {calc_leverage_ratio_rate:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Equity to Assets ratio for {symbol}: {calc_equity_to_assets_ratio_rate:.2f}{Style.RESET_ALL}")

    #יחסי יעילות תפעולית
    calc_receivables_ratio_rate=calc_receivables_ratio(balance_sheet,income_statement)
    calc_customers_ratio_rate = calc_days_sales_outstanding(balance_sheet, income_statement)
    inventory_ratio = calc_inventory_ratio(balance_sheet, income_statement)
    inventory_turnover_ratio = calc_inventory_turnover_ratio(balance_sheet, income_statement)
    inventory_days = calc_inventory_days(balance_sheet, income_statement)
    payables_days = calc_payables_days(balance_sheet, income_statement)
    print("\nיחסי יעילות תפעולית")
    print(f"{Fore.GREEN}Receivables ratio to Assets ratio for {symbol}: {calc_receivables_ratio_rate:.2f}{Style.RESET_ALL}")
    print( f"{Fore.GREEN}Customers days ratio ratio to Assets ratio for {symbol}: {calc_customers_ratio_rate:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Inventory ratio ratio to Assets ratio for {symbol}: {inventory_ratio:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Inventory turnover ratio ratio to Assets ratio for {symbol}: {inventory_turnover_ratio:.2f}{Style.RESET_ALL}")
    print( f"{Fore.GREEN}Inventory days ratio ratio to Assets ratio for {symbol}: {inventory_days:.2f}{Style.RESET_ALL}")
    print( f"{Fore.GREEN}Payables Days ratio ratio to Assets ratio for {symbol}: {payables_days:.2f}{Style.RESET_ALL}")


from colorama import Fore, Style


# פונקציה לעדכון צבעים בהתאם לערכים פיננסיים
def get_color_by_value(value, growth_threshold, stable_threshold):
    if value > growth_threshold:
        return Fore.GREEN  # צמיחה
    elif stable_threshold <= value <= growth_threshold:
        return Fore.YELLOW  # יציבות
    else:
        return Fore.RED  # דעיכה


def calc_Ratios_with_growth(stock, symbol):
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cash_flow = stock.cashflow

    # יחסי נזילות
    current_ratio_value = current_ratio(balance_sheet)
    quick_ratio_value = quick_ratio(balance_sheet)
    immediate_liquidity_ratio = calc_liquidity_ratio(balance_sheet)
    cashflow_to_sales_ratio = calc_cashflow_to_sales_ratio(cash_flow, income_statement)

    # הגדרת הספים (לדוגמה)
    growth_threshold = 1.5  # לדוגמה, ערך מעל זה הוא צמיחה
    stable_threshold = 1.0  # ערך בין זה לבין הסף העליון מצביע על יציבות

    # יחסי נזילות - צבע מותאם
    print("יחסי נזילות:")
    print(
        f"{get_color_by_value(current_ratio_value, growth_threshold, stable_threshold)}Current Ratio (יחס שוטף) for {symbol}: {current_ratio_value:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(quick_ratio_value, growth_threshold, stable_threshold)}Quick Ratio (יחס מהיר) for {symbol}: {quick_ratio_value:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(immediate_liquidity_ratio, growth_threshold, stable_threshold)}Immediate Liquidity Ratio (רמת נזילות מיידית) for {symbol}: {immediate_liquidity_ratio:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(cashflow_to_sales_ratio, growth_threshold, stable_threshold)}Cash Flow to Sales Ratio for {symbol}: {cashflow_to_sales_ratio:.2f}{Style.RESET_ALL}")

    # יחסי רווחיות
    net_profit_margin_value = calc_net_profit_margin(income_statement)
    operating_profit_margin_value = calc_operating_profit_margin(income_statement)
    ebitda_ratio = calc_ebitda_ratio(income_statement)
    roe_value = calc_roe(balance_sheet, income_statement)
    roa_value = calc_roa(balance_sheet, income_statement)

    # הדפסת יחסי רווחיות עם צבעים
    print("\nיחסי רווחיות")
    print(
        f"{get_color_by_value(net_profit_margin_value, growth_threshold, stable_threshold)}Net Profit Margin (שיעור הרווח הנקי) for {symbol}: {net_profit_margin_value:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(operating_profit_margin_value, growth_threshold, stable_threshold)}Operating Profit Margin (שיעור הרווח התפעולי) for {symbol}: {operating_profit_margin_value:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(ebitda_ratio, growth_threshold, stable_threshold)}EBITDA to Sales Ratio for {symbol}: {ebitda_ratio:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(roe_value, growth_threshold, stable_threshold)}ROE for {symbol}: {roe_value:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(roa_value, growth_threshold, stable_threshold)}ROA for {symbol}: {roa_value:.2f}{Style.RESET_ALL}")

    # יחסי מבני הון
    market_value_of_equity = stock.info['marketCap']
    leverage_ratio_value = calc_leverage_ratio(balance_sheet, market_value_of_equity)
    equity_to_assets_ratio_value = calc_equity_to_assets_ratio(balance_sheet)

    print("\nיחסי מבני הון")
    print(
        f"{get_color_by_value(leverage_ratio_value, growth_threshold, stable_threshold)}Leverage ratio for {symbol}: {leverage_ratio_value:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(equity_to_assets_ratio_value, growth_threshold, stable_threshold)}Equity to Assets ratio for {symbol}: {equity_to_assets_ratio_value:.2f}{Style.RESET_ALL}")

    # יחסי יעילות תפעולית
    receivables_ratio = calc_receivables_ratio(balance_sheet, income_statement)
    customers_ratio = calc_days_sales_outstanding(balance_sheet, income_statement)
    inventory_ratio = calc_inventory_ratio(balance_sheet, income_statement)
    inventory_turnover_ratio = calc_inventory_turnover_ratio(balance_sheet, income_statement)
    inventory_days = calc_inventory_days(balance_sheet, income_statement)
    payables_days = calc_payables_days(balance_sheet, income_statement)

    print("\nיחסי יעילות תפעולית")
    print(
        f"{get_color_by_value(receivables_ratio, growth_threshold, stable_threshold)}Receivables ratio to Assets ratio for {symbol}: {receivables_ratio:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(customers_ratio, growth_threshold, stable_threshold)}Customers days ratio for {symbol}: {customers_ratio:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(inventory_ratio, growth_threshold, stable_threshold)}Inventory ratio for {symbol}: {inventory_ratio:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(inventory_turnover_ratio, growth_threshold, stable_threshold)}Inventory turnover ratio for {symbol}: {inventory_turnover_ratio:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(inventory_days, growth_threshold, stable_threshold)}Inventory days for {symbol}: {inventory_days:.2f}{Style.RESET_ALL}")
    print(
        f"{get_color_by_value(payables_days, growth_threshold, stable_threshold)}Payables Days ratio for {symbol}: {payables_days:.2f}{Style.RESET_ALL}")


calc_Ratios_with_growth(stock,symbol)








