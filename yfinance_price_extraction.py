import yfinance as yf
from datetime import datetime

token_name = "ETH-USD"

dat = yf.Ticker(token_name)

df = dat.history(period='1y', interval='1d')
df.to_csv(f'prices_{token_name}_{datetime.now().date()}.csv', index=False)
#print(dat.get_history_metadata())