import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

# defining features and labels
X = np.array(df.drop(['label'],1))
Y = np.array(df['label'])

# preprocessing the data - scaling the features between -1 and 1
X = preprocessing.scale(X)

# creating training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# classifier
clf = LinearRegression()
# train the data
clf.fit(X_train, Y_train)
# get test score
accuracy = clf.score(X_test, Y_test)

print(accuracy)
