'''Write a program to construct a Bayesian network 
considering medical data. Use this model to 
demonstrate the diagnosis of heart patients using 
standard Heart Disease Data Set. You 
can use Java/Python ML library classes/API.'''

import pandas as pd

dataset = pd.read_csv('heart.csv')

X = dataset.iloc[:].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))