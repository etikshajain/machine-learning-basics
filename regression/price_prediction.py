import pandas as pd
import quandl
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

style.use('ggplot')

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
# df.dropna(inplace=True)

# defining features and labels
X = np.array(df.drop(['label'],1))

# preprocessing the data - scaling the features between -1 and 1
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace=True)

Y = np.array(df['label'])

# creating training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# classifier
clf = LinearRegression()
# train the data
clf.fit(X_train, Y_train)
# get test score
accuracy = clf.score(X_test, Y_test)

print(accuracy)

# predict outputs
forecast_set = clf.predict(X_lately)
print(forecast_set, forecast_out)

# graph
df['Forecast'] = np.nan #unpredicted data
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
