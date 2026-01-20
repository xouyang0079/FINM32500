import csv
from datetime import datetime
from models import MarketDataPoint

# Read market_data.csv (columns: timestamp, symbol, price) using the built-in csv module.
def load_data(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        datareader = csv.reader(f)
        next(datareader)
        for row in datareader:
            sample = MarketDataPoint(
                timestamp=datetime.fromisoformat(row[0]), 
                symbol=row[1],
                price=float(row[2]))

            data.append(sample)
    
    return data