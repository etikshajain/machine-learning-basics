import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pickle

# loading data frame
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace=True)

# label -> df['class'] => 2 for benign, 4 for malignant

# defining datasets
X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])

# normalising X
X = preprocessing.scale(X)

# split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

# classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

# saving the model
with open('LR.pickle', "wb") as f:
    pickle.dump(clf, f)

pickle_in = open('LR.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, Y_test)

print(accuracy)

# predict
example = np.array([[1,2,1,1,1,2,3,2,1]])
example = example.reshape(len(example),-1)
prediction = clf.predict(example)
print(prediction)