import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Settings
num_sap_accounts = 1000
cob_dates = ["2023-09-29", "2024-09-30"]
trade_start_date = datetime(2023, 1, 1)
trade_end_date = datetime(2024, 9, 30)

# Generate unique Trade IDs, keeping them common across COB dates
trade_dates = pd.date_range(trade_start_date, trade_end_date)
trade_ids = [f"TRADE-{i+1:04d}" for i in range(len(trade_dates))]

np.random.seed(42)  
sap_account_trade_map = {f"ACC-{i+1:04d}": np.random.choice(trade_ids, size=np.random.randint(2, 5), replace=False) 
                         for i in range(num_sap_accounts)}

# Create DataFrame for all combinations
data = []

for cob_date in cob_dates:
    for sap_account, account_trade_ids in sap_account_trade_map.items():
        for trade_id in account_trade_ids:
            # Assign a random trade date for each trade ID
            trade_date = np.random.choice(trade_dates)
            maturity_date = trade_date + timedelta(days=1)
            principal_amount = np.round(np.random.uniform(1, 10), 2) * 1e6  

            balance_amount = np.round(np.random.uniform(20000, 99000), 2)
            interest_rate = np.round(np.random.uniform(4.5, 5.8), 2)
            floating_rate = np.round(np.random.uniform(4.0, 6.0), 2)

            data.append([
                cob_date,
                sap_account,
                floating_rate,
                interest_rate,
                balance_amount,
                trade_date,
                maturity_date,
                trade_id,
                principal_amount
            ])

# Create DataFrame
columns = [
    "COB Date",
    "SAP Account",
    "Floating Rate",
    "Interest Rate",
    "Balance Amount",
    "Trade Date",
    "Maturity Date",
    "Trade ID",
    "Principal/Notional Amount"
]
df = pd.DataFrame(data, columns=columns)

# Display a sample of the data
df.head(10)
