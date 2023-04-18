import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')

# setting features
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'])*100;
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'])*100;

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# fill missing data
df.fillna(-99999, inplace=True)

# setting the label
forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df))) # we want to predict the price of 0.1*len(df) days later
df['label'] = df[forecast_col].shift(-forecast_out) # make the label column as the price of 0.1*len(df) days later
df.dropna(inplace=True)

print(df.tail())
