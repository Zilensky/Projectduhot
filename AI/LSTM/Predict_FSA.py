import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess dataset

lst={'Symbol': 0, 'Year': 2, 'Ordinary Shares Number': 3, 'Share Issued': 4, 'Net Debt': 5, 'Total Debt': 6, 'Tangible Book Value': 7, 'Invested Capital': 8, 'Working Capital': 9, 'Net Tangible Assets': 10, 'Capital Lease Obligations': 11, 'Common Stock Equity': 12, 'Total Capitalization': 13, 'Total Equity Gross Minority Interest': 14, 'Minority Interest': 15, 'Stockholders Equity': 16, 'Other Equity Interest': 17, 'Treasury Stock': 18, 'Retained Earnings': 19, 'Additional Paid In Capital': 20, 'Capital Stock': 21, 'Common Stock': 22, 'Total Liabilities Net Minority Interest': 23, 'Total Non Current Liabilities Net Minority Interest': 24, 'Non Current Deferred Taxes Liabilities': 25, 'Long Term Debt And Capital Lease Obligation': 26, 'Long Term Capital Lease Obligation': 27, 'Long Term Debt': 28, 'Current Liabilities': 29, 'Other Current Liabilities': 30, 'Current Debt And Capital Lease Obligation': 31, 'Current Capital Lease Obligation': 32, 'Current Debt': 33, 'Current Provisions': 34, 'Payables': 35, 'Other Payable': 36, 'Accounts Payable': 37, 'Total Assets': 38, 'Total Non Current Assets': 39, 'Other Non Current Assets': 40, 'Non Current Deferred Taxes Assets': 41, 'Long Term Equity Investment': 42, 'Investment Properties': 43, 'Goodwill And Other Intangible Assets': 44, 'Other Intangible Assets': 45, 'Goodwill': 46, 'Net PPE': 47, 'Gross PPE': 48, 'Construction In Progress': 49, 'Other Properties': 50, 'Current Assets': 51, 'Restricted Cash': 52, 'Prepaid Assets': 53, 'Inventory': 54, 'Finished Goods': 55, 'Work In Process': 56, 'Other Receivables': 57, 'Accounts Receivable': 58, 'Cash Cash Equivalents And Short Term Investments': 59, 'Other Short Term Investments': 60, 'Cash And Cash Equivalents': 61, 'Tax Effect Of Unusual Items': 62, 'Tax Rate For Calcs': 63, 'Normalized EBITDA': 64, 'Total Unusual Items': 65, 'Total Unusual Items Excluding Goodwill': 66, 'Net Income From Continuing Operation Net Minority Interest': 67, 'Reconciled Depreciation': 68, 'Reconciled Cost Of Revenue': 69, 'EBITDA': 70, 'EBIT': 71, 'Net Interest Income': 72, 'Interest Expense': 73, 'Interest Income': 74, 'Normalized Income': 75, 'Net Income From Continuing And Discontinued Operation': 76, 'Total Expenses': 77, 'Rent Expense Supplemental': 78, 'Total Operating Income As Reported': 79, 'Diluted Average Shares': 80, 'Basic Average Shares': 81, 'Diluted EPS': 82, 'Basic EPS': 83, 'Diluted NI Availto Com Stockholders': 84, 'Net Income Common Stockholders': 85, 'Otherunder Preferred Stock Dividend': 86, 'Net Income': 87, 'Minority Interests': 88, 'Net Income Including Noncontrolling Interests': 89, 'Net Income Continuous Operations': 90, 'Tax Provision': 91, 'Pretax Income': 92, 'Other Non Operating Income Expenses': 93, 'Special Income Charges': 94, 'Other Special Charges': 95, 'Net Non Operating Interest Income Expense': 96, 'Total Other Finance Cost': 97, 'Interest Expense Non Operating': 98, 'Interest Income Non Operating': 99, 'Operating Income': 100, 'Operating Expense': 101, 'Other Operating Expenses': 102, 'Depreciation And Amortization In Income Statement': 103, 'Depreciation Income Statement': 104, 'Selling General And Administration': 105, 'Selling And Marketing Expense': 106, 'General And Administrative Expense': 107, 'Rent And Landing Fees': 108, 'Gross Profit': 109, 'Cost Of Revenue': 110, 'Total Revenue': 111, 'Operating Revenue': 112, 'Free Cash Flow': 113, 'Repurchase Of Capital Stock': 114, 'Repayment Of Debt': 115, 'Issuance Of Debt': 116, 'Issuance Of Capital Stock': 117, 'Capital Expenditure': 118, 'End Cash Position': 119, 'Beginning Cash Position': 120, 'Effect Of Exchange Rate Changes': 121, 'Changes In Cash': 122, 'Financing Cash Flow': 123, 'Cash Dividends Paid': 124, 'Common Stock Dividend Paid': 125, 'Net Common Stock Issuance': 126, 'Common Stock Payments': 127, 'Common Stock Issuance': 128, 'Net Issuance Payments Of Debt': 129, 'Net Short Term Debt Issuance': 130, 'Net Long Term Debt Issuance': 131, 'Long Term Debt Payments': 132, 'Long Term Debt Issuance': 133, 'Investing Cash Flow': 134, 'Net Other Investing Changes': 135, 'Dividends Received Cfi': 136, 'Net Investment Purchase And Sale': 137, 'Sale Of Investment': 138, 'Purchase Of Investment': 139, 'Net Investment Properties Purchase And Sale': 140, 'Sale Of Investment Properties': 141, 'Purchase Of Investment Properties': 142, 'Net Business Purchase And Sale': 143, 'Purchase Of Business': 144, 'Net Intangibles Purchase And Sale': 145, 'Sale Of Intangibles': 146, 'Purchase Of Intangibles': 147, 'Net PPE Purchase And Sale': 148, 'Sale Of PPE': 149, 'Purchase Of PPE': 150, 'Operating Cash Flow': 151, 'Taxes Refund Paid': 152, 'Interest Received Cfo': 153, 'Interest Paid Cfo': 154, 'Change In Working Capital': 155, 'Change In Payable': 156, 'Change In Inventory': 157, 'Change In Receivables': 158, 'Other Non Cash Items': 159, 'Stock Based Compensation': 160, 'Deferred Tax': 161, 'Depreciation And Amortization': 162, 'Depreciation': 163, 'Gain Loss On Investment Securities': 164, 'Gain Loss On Sale Of PPE': 165, 'Net Income From Continuing Operations': 166, 'Fixed Assets Revaluation Reserve': 167, 'Other Non Current Liabilities': 168, 'Non Current Pension And Other Postretirement Benefit Plans': 169, 'Tradeand Other Payables Non Current': 170, 'Total Tax Payable': 171, 'Financial Assets': 172, 'Accumulated Depreciation': 173, 'Machinery Furniture Equipment': 174, 'Land And Improvements': 175, 'Properties': 176, 'Hedging Assets Current': 177, 'Raw Materials': 178, 'Taxes Receivable': 179, 'Allowance For Doubtful Accounts Receivable': 180, 'Gross Accounts Receivable': 181, 'Cash Equivalents': 182, 'Cash Financial': 183, 'Impairment Of Capital Assets': 184, 'Research And Development': 185, 'Net Other Financing Charges': 186, 'Short Term Debt Payments': 187, 'Short Term Debt Issuance': 188, 'Change In Other Current Assets': 189, 'Provisionand Write Offof Assets': 190, 'Pension And Employee Benefit Expense': 191, 'Employee Benefits': 192, 'Non Current Deferred Liabilities': 193, 'Non Current Deferred Revenue': 194, 'Current Deferred Liabilities': 195, 'Current Deferred Revenue': 196, 'Pensionand Other Post Retirement Benefit Plans Current': 197, 'Payables And Accrued Expenses': 198, 'Current Accrued Expenses': 199, 'Non Current Prepaid Assets': 200, 'Non Current Deferred Assets': 201, 'Non Current Accounts Receivable': 202, 'Leases': 203, 'Receivables': 204, 'Accrued Interest Receivable': 205, 'Other Income Expense': 206, 'Gain On Sale Of Security': 207, 'Other Gand A': 208, 'Salaries And Wages': 209, 'Cash Flow From Continuing Financing Activities': 210, 'Proceeds From Stock Option Exercised': 211, 'Cash Flow From Continuing Investing Activities': 212, 'Cash Flow From Continuing Operating Activities': 213, 'Change In Other Working Capital': 214, 'Change In Payables And Accrued Expense': 215, 'Change In Accrued Expense': 216, 'Change In Account Payable': 217, 'Change In Prepaid Assets': 218, 'Changes In Account Receivables': 219, 'Deferred Income Tax': 220, 'Depreciation Amortization Depletion': 221, 'Amortization Cash Flow': 222, 'Amortization Of Intangibles': 223, 'Operating Gains Losses': 224, 'Treasury Shares Number': 225, 'Derivative Product Liabilities': 226, 'Dividends Payable': 227, 'Investmentin Financial Assets': 228, 'Financial Assets Designatedas Fair Value Through Profitor Loss Total': 229, 'Investments In Other Ventures Under Equity Method': 230, 'Investmentsin Associatesat Cost': 231, 'Buildings And Improvements': 232, 'Net Income Discontinuous Operations': 233, 'Write Off': 234, 'Restructuring And Mergern Acquisition': 235, 'Sale Of Business': 236, 'Net Foreign Currency Exchange Gain Loss': 237, 'Interest Received Cfi': 238, 'Dividend Received Cfo': 239, 'Interest Paid Cff': 240, 'Gain Loss On Sale Of Business': 241, 'Long Term Provisions': 242, 'Current Deferred Taxes Liabilities': 243, 'Inventories Adjustments Allowances': 244, 'Gains Losses Not Affecting Retained Earnings': 245, 'Other Equity Adjustments': 246, 'Other Current Borrowings': 247, 'Defined Pension Benefit': 248, 'Investments And Advances': 249, 'Available For Sale Securities': 250, 'Securities Amortization': 251, 'Interest Paid Supplemental Data': 252, 'Income Tax Paid Supplemental Data': 253, 'Change In Other Current Liabilities': 254, 'Change In Interest Payable': 255, 'Amortization Of Securities': 256, 'Other Current Assets': 257, 'Amortization': 258, 'Assets Held For Sale Current': 259, 'Capital Expenditure Reported': 260, 'Earnings From Equity Interest Net Of Tax': 261, 'Gain On Sale Of Ppe': 262, 'Depreciation Amortization Depletion Income Statement': 263, 'Amortization Of Intangibles Income Statement': 264, 'Unrealized Gain Loss On Investment Securities': 265, 'Asset Impairment Charge': 266, 'Earnings Losses From Equity Investments': 267, 'Investmentsin Subsidiariesat Cost': 268, 'Other Inventories': 269, 'Other Cash Adjustment Outside Changein Cash': 270, 'Line Of Credit': 271, 'Dueto Related Parties Current': 272, 'Other Investments': 273, 'Held To Maturity Securities': 274, 'Preferred Securities Outside Stock Equity': 275, 'Commercial Paper': 276, 'Interest Payable': 277, 'Income Tax Payable': 278, 'Other Taxes': 279, 'Current Deferred Assets': 280, 'Preferred Stock Equity': 281, 'Preferred Stock': 282, 'Net Preferred Stock Issuance': 283, 'Preferred Stock Issuance': 284, 'Foreign Currency Translation Adjustments': 285, 'Gain On Sale Of Business': 286, 'Earnings From Equity Interest': 287, 'Insurance And Claims': 288, 'Net Policyholder Benefits And Claims': 289, 'Net Income Extraordinary': 290, 'Non Current Note Receivables': 291, 'Duefrom Related Parties Current': 292, 'Loans Receivable': 293, 'Provision For Doubtful Accounts': 294, 'Average Dilution Earnings': 295, 'Cash From Discontinued Operating Activities': 296, 'Investmentsin Joint Venturesat Cost': 297, 'Non Current Accrued Expenses': 298, 'Receivables Adjustments Allowances': 299, 'Other Cash Adjustment Inside Changein Cash': 300, 'Preferred Stock Dividends': 301, 'Trading Securities': 302, 'Cash Cash Equivalents And Federal Funds Sold': 303, 'Dividend Paid Cfo': 304, 'Liabilities Heldfor Sale Non Current': 305, 'Minimum Pension Liabilities': 306, 'Preferred Stock Dividend Paid': 307, 'Preferred Shares Number': 308, 'Notes Receivable': 309, 'Preferred Stock Payments': 310, 'Current Deferred Taxes Assets': 311, 'Current Notes Payable': 312, 'Change In Tax Payable': 313, 'Change In Income Tax Payable': 314, 'Cash Flow From Discontinued Operation': 315, 'Cash From Discontinued Financing Activities': 316, 'Cash From Discontinued Investing Activities': 317, 'Unrealized Gain Loss': 318}

def extract_feature_name_by_id(lst, feature_id):
    # Check if the feature ID exists in the dictionary
    if feature_id in lst.values():
        # Find the key (feature name) corresponding to the feature ID
        feature_name = [k for k, v in lst.items() if v == feature_id]
        return feature_name[0]  # Return the first match
    else:
        raise ValueError(f"Feature ID {feature_id} not found in the list.")


def load_and_preprocess_data(file_path, save_json_path='columns.json'):
    """
    Loads and preprocesses financial data for sequential modeling.
    Dynamically extracts column names and saves them as JSON for feature comparison.
    Processes into input-output sequences for training and scales data between 0 and 1.

    :param file_path: Path to the Excel file containing the data.
    :return: DataLoader, MinMaxScaler instance, List of symbols
    """
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    import json

    # Load the data
    df = pd.read_excel(file_path)
    print("Dataframe loaded successfully.")
    print(df)
    # Ensure column names are unique
    if df.columns.duplicated().any():
        print("Duplicate column names detected. Renaming duplicates...")
        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names()
        print("Renamed columns:", df.columns.tolist())

    # Verify required columns
    required_columns = {'Symbol', 'Year'}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select numeric columns and keep Symbol and Year
    numeric_columns = df.select_dtypes(include=np.number).columns
    df = df[['Symbol', 'Year'] + list(numeric_columns)]
    print(f"Filtered DataFrame shape: {df.shape}")

    # Drop rows with missing numeric data
    df = df.dropna(subset=numeric_columns, how='all')
    print(f"Dataframe shape after dropping rows with all NaN numeric values: {df.shape}")

    # Define required years and initialize containers
    required_years = [2019, 2020, 2021, 2022]
    sequential_data = []
    valid_symbols = []

    # Initialize MinMaxScaler

    # Save column-to-feature-ID mapping to JSON
    print(df.columns.tolist())
    df_columns = {col: idx for idx, col in enumerate(df.columns[3:])}
    with open('columns.json', 'w') as f:
        json.dump(df_columns, f)
    with open('columns.json', 'r') as f:
        print(f.read())
    # Group by symbol and process each symbol's data
    grouped = df.groupby('Symbol')
    for symbol, group in grouped:
        # Ensure no duplicate years and sort by year
        group = group.loc[:, ~group.columns.duplicated()]  # Remove any duplicate columns
        group = group.drop_duplicates(subset=['Year']).sort_values(by='Year')

        # Check if all required years are present
        if not set(required_years).issubset(group['Year'].values):
            print(f"Skipping symbol {symbol} due to missing years.")
            continue

        # Extract numeric data for required years
        try:
            year_data = []
            for year in required_years:
                row = group[group['Year'] == year]
                if row.empty:
                    raise ValueError(f"Missing data for year {year}")
                numeric_data = row[numeric_columns].iloc[0].to_numpy() if year in required_years else np.zeros(
                    len(numeric_columns))
                year_data.append(np.nan_to_num(numeric_data, nan=0.0))  # Handle NaN values

            # Convert year_data to NumPy array
            year_data = np.array(year_data)

            # Scale the numeric data
          #  year_data_scaled = scaler.fit_transform(year_data)
            sequential_data.append(year_data)
            valid_symbols.append(symbol)
        except Exception as e:
            print(f"Skipping symbol {symbol} due to data error: {e}")

    # Ensure valid data exists
    if not sequential_data:
        raise RuntimeError("No valid symbol sequences found after preprocessing. Check your dataset.")

    # Convert to tensors
    sequential_data = np.array(sequential_data, dtype=np.float32)
    inputs = torch.tensor(sequential_data, dtype=torch.float32)

    outputs = inputs.clone()

    print(f"Processed {len(valid_symbols)} valid symbol sequences.")
    print(f"Input tensor shape: {inputs.shape}")

    # Create DataLoader
    dataset = TensorDataset(inputs, outputs)
    return DataLoader(dataset, batch_size=32, shuffle=True), None, valid_symbols



# Define the LSTM Autoencoder class
import torch
import torch.nn as nn


class LSTM_SequencePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, output_length):
        """
        LSTM Model to directly predict the next set of years based on input sequence.

        :param input_size: Number of input features.
        :param hidden_size: Number of units in the hidden layer.
        :param sequence_length: Length of the input sequence (e.g., 4 years).
        :param output_length: Length of the output sequence to predict (e.g., 4 years).
        """
        super(LSTM_SequencePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size * output_length)  # Directly map to future years

        # Store sequence and output lengths
        self.input_size = input_size
        self.output_length = output_length

    def forward(self, x):
        """
        Forward pass through the LSTM for direct prediction.

        :param x: Input tensor of shape (batch_size, sequence_length, input_size).
        :param sequence_length: Length of the input sequence (e.g., 4 years: 2019-2022).
        :param output_length: Length of the output sequence to predict (e.g., 5 years: 2023-2027).
"""
        # Process through LSTM for 4 years input sequence
        _, (hidden, _) = self.lstm(x)

        # Decode hidden state to predict the next sequence
        output = self.fc(hidden[-1])  # Use only the last hidden state

        # Reshape to (batch_size, output_length, input_size)
        predictions = output.view(-1, self.output_length, self.input_size)
        return predictions


# Training function
def train_model(model, data_loader, num_epochs, criterion, optimizer, sequence_length=4, output_length=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
import pandas as pd
import json
import torch

def convert_model_to_dataframe(model_outputs, columns_file='columns.json'):
    """
    Converts the model's output tensor to a DataFrame based on the columns.json mapping.

    :param model_outputs: The output tensor from the model (shape: [batch_size, output_length, feature_size]).
    :param columns_file: Path to the JSON file that contains the column mappings.
    :return: A DataFrame with predictions for the features.
    """
    import pandas as pd
    import json
    import torch

    # Load the column-name-to-index mapping
    with open(columns_file, 'r') as f:
        columns_mapping = json.load(f)

    # Reverse the mapping to get feature names based on index
    id_to_column = {v: k for k, v in columns_mapping.items()}

    # Convert model_outputs tensor to numpy array if necessary
    if isinstance(model_outputs, torch.Tensor):
        model_outputs = model_outputs.cpu().detach().numpy()

    # Validate and filter feature indices
    valid_feature_indices = [
        idx for idx in range(model_outputs.shape[2]) if idx in id_to_column
    ]
    if not valid_feature_indices:
        raise ValueError("No valid feature indices found in the model outputs.")

    feature_names = [id_to_column[idx] for idx in valid_feature_indices]

    # Extract data for only the valid feature indices
    flattened_data = []
    for batch_idx in range(model_outputs.shape[0]):
        for time_idx in range(model_outputs.shape[1]):
            flattened_data.append(model_outputs[batch_idx, time_idx, valid_feature_indices])

    # Create a DataFrame with the valid feature names as columns
    predictions_df = pd.DataFrame(flattened_data, columns=feature_names)

    print(predictions_df)
    return predictions_df

# Prediction and visualization
def predict_and_analyze(
        model, data_loader, symbols, scaler, feature_name="Total Revenue",
        sequence_length=4, output_length=5
):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import mean_squared_error

    # Ensure the model is in evaluation mode
    model.eval()

    # Retrieve feature index
    lst_symbols = lst
    if feature_name not in lst_symbols:
        raise ValueError(f"Feature name '{feature_name}' not found in lst_symbols.")
    feature_idx = lst_symbols[feature_name]

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            # Move inputs and targets to the correct device (GPU/CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate model predictions
            outputs = model(inputs).cpu().numpy()  # Predicted values (2023-2027)
            actual = inputs# Actual values (2019-2022)
            predictions_df = convert_model_to_dataframe(outputs, columns_file='columns.json')
            # Inverse-transform scaling – actual
            print(predictions_df)


            # Break after first symbol (remove this to process all symbols)
            break

# Main function
def main():
    file_path = 'combined_financial_data_all_stocks.xlsx'
    data_loader, scaler, symbols = load_and_preprocess_data(file_path)

    # Correctly compute input size
    input_size = data_loader.dataset.tensors[0].shape[2]  # Number of features
    hidden_size = 64
    sequence_length = 4  # 4 years: 2019 to 2022
    output_length = 4 # Predict 5 years: 2023 to 2027

    model = LSTM_SequencePredictor(input_size, hidden_size, sequence_length, output_length).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)

    train_model(model, data_loader, num_epochs=50, criterion=criterion, optimizer=optimizer)
    predict_and_analyze(model, data_loader,  symbols,scaler)

if __name__ == '__main__':
    main()
