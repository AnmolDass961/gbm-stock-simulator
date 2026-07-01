import pandas as pd
import yfinance as yf

def fetch_data(ticker,start_date,end_date)->pd.Series:
    data=yf.download(ticker,start=start_date,end=end_date,auto_adjust=True,progress=False)
    if data.empty:
        raise ValueError(f'No data found for ticker {ticker} between {start_date} and {end_date}.')
    return data['Close'].squeeze()
